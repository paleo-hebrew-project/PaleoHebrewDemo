#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Paleo-Hebrew Epigraphy Pipeline (Gradio)

Stable + recolored version for Hugging Face Spaces:
- blue/cyan/teal UI palette instead of orange;
- blue/cyan bbox palette instead of default Plotly orange-first palette;
- protects lazy model loading with a lock;
- protects optional translator loading with a lock;
- serializes heavy inference callbacks with a shared lock and Gradio concurrency_id;
- disables Gradio SSR when supported;
- disables Run buttons while inference is running to reduce accidental double clicks.
"""

import os

# Must be set before importing gradio as much as possible.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

# HF Spaces / Gradio 6 can enable SSR by default. For this app SSR produced many
# transient static asset 404s in logs, so keep it disabled unless explicitly overridden.
if os.getenv("PALEO_ENABLE_GRADIO_SSR", "0") != "1":
    os.environ["GRADIO_SSR_MODE"] = "False"

import gc
import io
import json
import time
import base64
import inspect
import tempfile
import traceback
import logging
import threading
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import gradio as gr
import torch
from huggingface_hub import hf_hub_download

try:
    from huggingface_hub.utils import disable_progress_bars

    disable_progress_bars()
except Exception:
    pass

import plotly.graph_objects as go


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("paleo_demo")


# -----------------------------------------------------------------------------
# Configuration via environment variables
# -----------------------------------------------------------------------------

DET_REPO_ID = os.getenv("DET_REPO_ID", "mr3vial/paleo-hebrew-yolo")
DET_FILENAME = os.getenv("DET_FILENAME", "best.onnx")
DET_FALLBACK_PT = os.getenv("DET_FALLBACK_PT", "best.pt")

CLS_REPO_ID = os.getenv("CLS_REPO_ID", "mr3vial/paleo-hebrew-convnext")
CLS_WEIGHTS = os.getenv("CLS_WEIGHTS", "convnext_weights.pt")
CLS_CLASSES = os.getenv("CLS_CLASSES", "classes.json")
CLS_MODEL_NAME = os.getenv("CLS_MODEL_NAME", "convnext_large")

MT5_BOX2HE_REPO = os.getenv(
    "MT5_BOX2HE_REPO", "mr3vial/paleo-hebrew-mt5-post-ocr-processing"
)
MT5_BOX2EN_REPO = os.getenv("MT5_BOX2EN_REPO", "mr3vial/paleo-hebrew-mt5-translate")

FEEDBACK_PATH = os.getenv("FEEDBACK_PATH", "/tmp/feedback.jsonl")

EXAMPLES_DIR = Path(__file__).parent / "examples"

IMG_HEIGHT = 420

QUEUE_MAX_SIZE = int(os.getenv("QUEUE_MAX_SIZE", "4"))
DEFAULT_CONCURRENCY_LIMIT = int(os.getenv("DEFAULT_CONCURRENCY_LIMIT", "1"))
MODEL_CONCURRENCY_ID = "single_model_gpu_queue"


# -----------------------------------------------------------------------------
# Locks
# -----------------------------------------------------------------------------

_LOAD_LOCK = threading.Lock()
_TRANSLATOR_LOCK = threading.Lock()

# Extra protection in case some Gradio/API path bypasses the intended queue
# isolation. This prevents simultaneous YOLO/ConvNeXt/mT5 calls on a small Space.
_INFER_LOCK = threading.Lock()


# -----------------------------------------------------------------------------
# Visual palette
# -----------------------------------------------------------------------------

BBOX_PALETTE = [
    "#0891b2",  # cyan-600
    "#0284c7",  # sky-600
    "#0f766e",  # teal-700
    "#2563eb",  # blue-600
    "#06b6d4",  # cyan-500
    "#0369a1",  # sky-700
    "#14b8a6",  # teal-500
    "#1d4ed8",  # blue-700
    "#22d3ee",  # cyan-400
    "#38bdf8",  # sky-400
]


# -----------------------------------------------------------------------------
# Translator choices (Hebrew -> English) for the he_then_en mode
# -----------------------------------------------------------------------------

TRANSLATOR_CHOICES = [
    ("None (use direct box→en)", "none"),
    ("OPUS-MT he→en (Helsinki-NLP/opus-mt-tc-big-he-en)", "opus"),
    ("NLLB-200 distilled 600M (he→en) [CC-BY-NC]", "nllb"),
    ("M2M100 418M (he→en)", "m2m100"),
]

_TRANSLATOR_CACHE: Dict[str, Tuple[Any, Any, str]] = {}


# -----------------------------------------------------------------------------
# Hebrew normalization (final forms -> base forms)
# -----------------------------------------------------------------------------

FINAL_MAP: Dict[str, str] = {
    "ך": "כ",
    "ם": "מ",
    "ן": "נ",
    "ף": "פ",
    "ץ": "צ",
}


def normalize_hebrew_letter(ch: str) -> str:
    return FINAL_MAP.get(ch, ch)


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------


def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def normalize_spaces(s: str) -> str:
    return " ".join((s or "").strip().split())


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def safe_int(x: float, default: int = 0) -> int:
    try:
        return int(round(float(x)))
    except Exception:
        return default


def supports_param(fn: Any, name: str) -> bool:
    try:
        return name in inspect.signature(fn).parameters
    except Exception:
        return False


def kwargs_supported_by(fn: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in kwargs.items() if supports_param(fn, k)}


def sort_boxes_reading_order(boxes_xyxy: List[List[float]], rtl: bool = True) -> List[int]:
    """Heuristic reading order: group by y (lines), sort by x within line."""
    if not boxes_xyxy:
        return []

    centers: List[Tuple[int, float, float]] = []
    heights: List[float] = []

    for i, (x1, y1, x2, y2) in enumerate(boxes_xyxy):
        centers.append((i, 0.5 * (x1 + x2), 0.5 * (y1 + y2)))
        heights.append(max(1.0, y2 - y1))

    line_tol = float(np.median(heights)) * 0.6

    centers_sorted = sorted(centers, key=lambda t: t[2])
    lines: List[List[Tuple[int, float, float]]] = []

    for item in centers_sorted:
        if not lines:
            lines.append([item])
            continue

        _, _, yc = item
        prev_line = lines[-1]
        prev_y = float(np.mean([p[2] for p in prev_line]))

        if abs(yc - prev_y) <= line_tol:
            prev_line.append(item)
        else:
            lines.append([item])

    idxs: List[int] = []
    for line in lines:
        line_sorted = sorted(line, key=lambda t: t[1], reverse=rtl)
        idxs.extend([i for i, _, _ in line_sorted])

    return idxs


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------


@torch.inference_mode()
def load_detector() -> Tuple[Any, str]:
    from ultralytics import YOLO  # lazy import

    try:
        det_path = hf_hub_download(repo_id=DET_REPO_ID, filename=DET_FILENAME)
        try:
            return YOLO(det_path, task="detect"), f"{DET_REPO_ID}/{DET_FILENAME}"
        except TypeError:
            return YOLO(det_path), f"{DET_REPO_ID}/{DET_FILENAME}"
    except Exception:
        log.exception("Failed to load ONNX detector; trying PT fallback.")
        pt_path = hf_hub_download(repo_id=DET_REPO_ID, filename=DET_FALLBACK_PT)
        try:
            return YOLO(pt_path, task="detect"), f"{DET_REPO_ID}/{DET_FALLBACK_PT} (PT fallback)"
        except TypeError:
            return YOLO(pt_path), f"{DET_REPO_ID}/{DET_FALLBACK_PT} (PT fallback)"


@torch.inference_mode()
def load_classifier() -> Tuple[Any, List[str], str]:
    import timm  # lazy import

    device = get_device()
    classes_path = hf_hub_download(repo_id=CLS_REPO_ID, filename=CLS_CLASSES)
    raw: List[Any] = []

    try:
        with open(classes_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict) and "classes" in data:
            raw = list(data["classes"])
        elif isinstance(data, dict):
            raw = list(data.values())
        elif isinstance(data, list):
            raw = data
    except Exception:
        with open(classes_path, "r", encoding="utf-8") as f:
            raw = [ln.strip() for ln in f if ln.strip()]

    letters: List[str] = []
    for ln in raw:
        val = str(ln).strip()
        val = val.split("_")[-1] if "_" in val else val
        val = val.strip('"\' ,')
        if val:
            letters.append(val)

    weights_path = hf_hub_download(repo_id=CLS_REPO_ID, filename=CLS_WEIGHTS)
    ckpt = torch.load(weights_path, map_location="cpu")
    state = ckpt.get("model", ckpt.get("state_dict", ckpt))

    if not isinstance(state, dict):
        raise RuntimeError(f"Bad checkpoint format: type(state)={type(state)}")

    new_state: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        nk = k
        for pref in ("module.", "model."):
            if nk.startswith(pref):
                nk = nk[len(pref) :]
        new_state[nk] = v

    if "head.fc.weight" not in new_state:
        raise RuntimeError("Checkpoint is missing head.fc.weight; cannot infer number of classes.")

    num_classes_ckpt = int(new_state["head.fc.weight"].shape[0])

    if len(letters) != num_classes_ckpt:
        log.warning(
            "classes.json has %d labels but checkpoint expects %d classes; truncating/padding.",
            len(letters),
            num_classes_ckpt,
        )
        if len(letters) > num_classes_ckpt:
            letters = letters[:num_classes_ckpt]
        else:
            letters = letters + [f"cls_{i:02d}" for i in range(len(letters), num_classes_ckpt)]

    model = timm.create_model(CLS_MODEL_NAME, pretrained=False, num_classes=num_classes_ckpt)
    model.load_state_dict(new_state, strict=False)
    model.eval().to(device)

    return model, letters, f"{CLS_REPO_ID}/{CLS_WEIGHTS} ({CLS_MODEL_NAME}, C={num_classes_ckpt})"


@torch.inference_mode()
def load_mt5(repo_id: str) -> Tuple[Any, Any]:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    device = get_device()
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32

    tok = AutoTokenizer.from_pretrained(repo_id, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(repo_id, torch_dtype=dtype).to(device).eval()

    return tok, model


@torch.inference_mode()
def mt5_generate(tok: Any, model: Any, text: str, max_new_tokens: int = 128) -> str:
    device = get_device()
    inp = tok([text], return_tensors="pt", truncation=True, max_length=2048).to(device)
    out = model.generate(**inp, max_new_tokens=max_new_tokens, num_beams=4, do_sample=False)
    return normalize_spaces(tok.batch_decode(out, skip_special_tokens=True)[0])


# -----------------------------------------------------------------------------
# Hebrew -> English translators (optional, only used in he_then_en mode)
# -----------------------------------------------------------------------------


@torch.inference_mode()
def get_he2en_translator(kind: str) -> Tuple[Any, Any, str]:
    global _TRANSLATOR_CACHE

    kind = (kind or "none").strip().lower()

    if kind in _TRANSLATOR_CACHE:
        return _TRANSLATOR_CACHE[kind]

    with _TRANSLATOR_LOCK:
        if kind in _TRANSLATOR_CACHE:
            return _TRANSLATOR_CACHE[kind]

        # Keep only one optional translator in memory at a time.
        _TRANSLATOR_CACHE.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        device = get_device()

        if kind == "opus":
            from transformers import MarianMTModel, MarianTokenizer

            repo = "Helsinki-NLP/opus-mt-tc-big-he-en"
            tok = MarianTokenizer.from_pretrained(repo)
            model = MarianMTModel.from_pretrained(repo).to(device).eval()
        elif kind == "nllb":
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            repo = "facebook/nllb-200-distilled-600M"
            tok = AutoTokenizer.from_pretrained(repo, src_lang="heb_Hebr")
            model = AutoModelForSeq2SeqLM.from_pretrained(repo).to(device).eval()
        elif kind == "m2m100":
            from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

            repo = "facebook/m2m100_418M"
            tok = M2M100Tokenizer.from_pretrained(repo)
            tok.src_lang = "he"
            model = M2M100ForConditionalGeneration.from_pretrained(repo).to(device).eval()
        else:
            _TRANSLATOR_CACHE[kind] = (None, None, "none")
            return _TRANSLATOR_CACHE[kind]

        _TRANSLATOR_CACHE[kind] = (tok, model, kind)
        return _TRANSLATOR_CACHE[kind]


@torch.inference_mode()
def translate_he_to_en(text_he: str, kind: str) -> str:
    if kind == "none":
        return ""

    tok, model, kind = get_he2en_translator(kind)
    device = get_device()

    if kind == "opus":
        batch = tok([text_he], return_tensors="pt", padding=True, truncation=True).to(device)
        out = model.generate(**batch, max_new_tokens=128, num_beams=4)
        return normalize_spaces(tok.batch_decode(out, skip_special_tokens=True)[0])

    if kind == "nllb":
        batch = tok([text_he], return_tensors="pt", truncation=True, max_length=512).to(device)
        out = model.generate(
            **batch,
            forced_bos_token_id=tok.convert_tokens_to_ids("eng_Latn"),
            max_new_tokens=128,
            num_beams=4,
        )
        return normalize_spaces(tok.batch_decode(out, skip_special_tokens=True)[0])

    if kind == "m2m100":
        batch = tok([text_he], return_tensors="pt", truncation=True, max_length=512).to(device)
        out = model.generate(
            **batch,
            forced_bos_token_id=tok.get_lang_id("en"),
            max_new_tokens=128,
            num_beams=4,
        )
        return normalize_spaces(tok.batch_decode(out, skip_special_tokens=True)[0])

    return ""


# -----------------------------------------------------------------------------
# Core pipeline
# -----------------------------------------------------------------------------


@dataclass
class Loaded:
    det: Any
    det_name: str
    cls: Any
    cls_letters: List[str]
    cls_name: str
    mt5_he_tok: Any
    mt5_he: Any
    mt5_en_tok: Any
    mt5_en: Any


LOADED: Optional[Loaded] = None


def ensure_loaded() -> Loaded:
    """Load all core models exactly once, even under concurrent cold-start requests."""
    global LOADED

    if LOADED is not None:
        return LOADED

    with _LOAD_LOCK:
        if LOADED is not None:
            return LOADED

        log.info("Loading core models...")
        det, det_name = load_detector()
        cls, letters, cls_name = load_classifier()
        he_tok, he_model = load_mt5(MT5_BOX2HE_REPO)
        en_tok, en_model = load_mt5(MT5_BOX2EN_REPO)

        LOADED = Loaded(
            det=det,
            det_name=det_name,
            cls=cls,
            cls_letters=letters,
            cls_name=cls_name,
            mt5_he_tok=he_tok,
            mt5_he=he_model,
            mt5_en_tok=en_tok,
            mt5_en=en_model,
        )

        log.info(
            "Loaded models: detector=%s classifier=%s box2he=%s box2en=%s",
            det_name,
            cls_name,
            MT5_BOX2HE_REPO,
            MT5_BOX2EN_REPO,
        )

        return LOADED


@torch.inference_mode()
def yolo_predict(
    det_model: Any,
    pil: Image.Image,
    conf: float,
    iou: float,
    max_det: int,
) -> Tuple[List[List[float]], List[float]]:
    with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as f:
        pil.save(f.name)
        res = det_model.predict(
            source=f.name,
            conf=float(conf),
            iou=float(iou),
            max_det=int(max_det),
            verbose=False,
        )

    r0 = res[0]
    if r0.boxes is None or len(r0.boxes) == 0:
        return [], []

    xyxy = r0.boxes.xyxy.detach().float().cpu().numpy().tolist()
    scores = r0.boxes.conf.detach().float().cpu().numpy().tolist()

    return xyxy, scores


@torch.inference_mode()
def classify_crops(
    cls_model: Any,
    letters: List[str],
    pil: Image.Image,
    boxes_xyxy: List[List[float]],
    pad: float,
    topk: int,
) -> Tuple[List[List[Tuple[str, float]]], List[str], List[Tuple[Image.Image, str]]]:
    device = get_device()
    W, H = pil.size
    crops: List[Image.Image] = []

    for (x1, y1, x2, y2) in boxes_xyxy:
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        p = int(round(max(w, h) * float(pad)))

        a = max(0, safe_int(x1) - p)
        b = max(0, safe_int(y1) - p)
        c = min(W, safe_int(x2) + p)
        d = min(H, safe_int(y2) + p)

        crop = pil.crop((a, b, c, d)).resize((96, 96), Image.BICUBIC)
        crops.append(crop)

    if not crops:
        return [], [], []

    def preprocess(img: Image.Image) -> torch.Tensor:
        arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return (t - mean) / std

    batch = torch.stack([preprocess(c) for c in crops], dim=0).to(device)
    logits = cls_model(batch)
    probs = torch.softmax(logits, dim=-1).detach().float().cpu().numpy()

    topk_list: List[List[Tuple[str, float]]] = []
    top1_letters: List[str] = []

    for i in range(len(crops)):
        p = probs[i]
        idx = np.argsort(-p)[: max(1, int(topk))]

        row: List[Tuple[str, float]] = []
        for j in idx:
            lab = normalize_hebrew_letter(letters[j])
            row.append((lab, float(p[j])))

        topk_list.append(row)
        top1_letters.append(row[0][0] if row else "?")

    crop_gallery = [(crops[i], f"{i + 1:02d}: {top1_letters[i]}") for i in range(len(crops))]
    return topk_list, top1_letters, crop_gallery


# -----------------------------------------------------------------------------
# mT5 source serialization
# -----------------------------------------------------------------------------


def _compute_box_normalizer_from_boxes(boxes: List[List[float]]) -> Tuple[float, float, float, float]:
    xs = [b[0] for b in boxes] + [b[2] for b in boxes]
    ys = [b[1] for b in boxes] + [b[3] for b in boxes]

    minx = float(min(xs)) if xs else 0.0
    maxx = float(max(xs)) if xs else 1.0
    miny = float(min(ys)) if ys else 0.0
    maxy = float(max(ys)) if ys else 1.0

    scalex = max(1e-6, maxx - minx)
    scaley = max(1e-6, maxy - miny)

    return minx, miny, scalex, scaley


def build_source_text_mt5(
    *,
    pil: Image.Image,
    boxes_xyxy: List[List[float]],
    det_scores: List[float],
    topk_list: List[List[Tuple[str, float]]],
    coord_norm: str,
    order_mode: str,
    rtl: bool,
) -> str:
    """Serialize boxes+cls into the format mT5 was trained on."""

    coord_norm = (coord_norm or "boxes").strip().lower()
    if coord_norm not in ("boxes", "det", "none"):
        coord_norm = "boxes"

    order_mode = (order_mode or "reading").strip().lower()
    if order_mode not in ("reading", "detector"):
        order_mode = "reading"

    if not boxes_xyxy:
        return "[BOXES_CLS_V3]\n" + "n=0 coord_norm=none"

    if order_mode == "reading":
        idxs = sort_boxes_reading_order(boxes_xyxy, rtl=rtl)
    else:
        idxs = list(range(len(boxes_xyxy)))

    W, H = pil.size

    if coord_norm == "boxes":
        minx, miny, scalex, scaley = _compute_box_normalizer_from_boxes(boxes_xyxy)
    elif coord_norm == "det":
        minx, miny, scalex, scaley = 0.0, 0.0, float(W), float(H)
    else:
        minx, miny, scalex, scaley = 0.0, 0.0, 1.0, 1.0

    lines = ["[BOXES_CLS_V3]", f"n={len(idxs)} coord_norm={coord_norm}"]

    for j, i in enumerate(idxs, start=1):
        x1, y1, x2, y2 = boxes_xyxy[i]
        sc = det_scores[i] if i < len(det_scores) else 1.0

        if coord_norm != "none":
            x1 = (x1 - minx) / scalex
            x2 = (x2 - minx) / scalex
            y1 = (y1 - miny) / scaley
            y2 = (y2 - miny) / scaley

        xa, xb = (x1, x2) if x1 <= x2 else (x2, x1)
        ya, yb = (y1, y2) if y1 <= y2 else (y2, y1)

        w = max(1e-6, xb - xa)
        h = max(1e-6, yb - ya)
        xc = 0.5 * (xa + xb)
        yc = 0.5 * (ya + yb)

        cands = topk_list[i] if i < len(topk_list) else []
        cls_str = " ".join([f"{ch}:{p:.3f}" for ch, p in cands]) if cands else "?"

        lines.append(
            f"{j:02d} x1={xa:.4f} y1={ya:.4f} x2={xb:.4f} y2={yb:.4f} "
            f"w={w:.4f} h={h:.4f} xc={xc:.4f} yc={yc:.4f} "
            f"score={sc:.3f} | cls {cls_str}"
        )

    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Plotly overlay (hover tooltips on bboxes)
# -----------------------------------------------------------------------------


def _pil_to_data_uri(pil: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    pil.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/{fmt.lower()};base64,{b64}"


def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    s = hex_color.lstrip("#")
    return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)


def make_bbox_figure(
    pil: Image.Image,
    boxes_xyxy: List[List[float]],
    labels: Optional[List[str]] = None,
    det_scores: Optional[List[float]] = None,
    topk_list: Optional[List[List[Tuple[str, float]]]] = None,
    height: int = IMG_HEIGHT,
) -> go.Figure:
    W, H = pil.size
    fig = go.Figure()

    fig.add_layout_image(
        dict(
            source=_pil_to_data_uri(pil),
            xref="x",
            yref="y",
            x=0,
            y=0,
            sizex=W,
            sizey=H,
            sizing="stretch",
            layer="below",
        )
    )

    for i, (x1, y1, x2, y2) in enumerate(boxes_xyxy):
        color_hex = BBOX_PALETTE[i % len(BBOX_PALETTE)]
        r, g, b = _hex_to_rgb(color_hex)

        lab = labels[i] if labels and i < len(labels) else f"{i + 1:02d}"
        sc = det_scores[i] if det_scores and i < len(det_scores) else None

        topk_str = ""
        if topk_list and i < len(topk_list) and topk_list[i]:
            topk_str = ", ".join([f"{c}:{p:.3f}" for c, p in topk_list[i]])

        hover_lines = [f"#{i + 1:02d}", f"top1: {lab}"]
        if sc is not None:
            hover_lines.append(f"det_conf: {sc:.3f}")
        if topk_str:
            hover_lines.append(f"topk: {topk_str}")

        xs = [x1, x2, x2, x1, x1]
        ys = [y1, y1, y2, y2, y1]

        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                fill="toself",
                line=dict(width=3, color=f"rgb({r},{g},{b})"),
                fillcolor=f"rgba({r},{g},{b},0.22)",
                hoverinfo="text",
                text="<br>".join(hover_lines),
                showlegend=False,
            )
        )

    fig.update_xaxes(visible=False, range=[0, W], constrain="domain")
    fig.update_yaxes(visible=False, range=[H, 0], scaleanchor="x")
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=30, b=0), hovermode="closest")

    return fig


# -----------------------------------------------------------------------------
# UI callbacks
# -----------------------------------------------------------------------------


def _run_pipeline_tab_impl(
    pil: Optional[Image.Image],
    conf: float,
    iou: float,
    max_det: int,
    crop_pad: float,
    topk_k: int,
    rtl: bool,
    output_mode: str,
    he2en_kind: str,
    coord_norm: str,
    order_mode: str,
) -> Tuple[
    Optional[Any],
    List[Tuple[Image.Image, str]],
    str,
    str,
    str,
    str,
    List[List[Tuple[str, float]]],
]:
    if pil is None:
        return None, [], "", "", json.dumps({"error": "No input image"}, ensure_ascii=False, indent=2), "", []

    pil = pil.convert("RGB")
    L = ensure_loaded()

    boxes, det_scores = yolo_predict(L.det, pil, conf, iou, max_det)

    if not boxes:
        fig = make_bbox_figure(pil, [], height=IMG_HEIGHT)
        dbg = {"ts": now_ts(), "detector": L.det_name, "n_boxes": 0}
        return (
            fig,
            [],
            "No boxes detected.",
            "No boxes detected.",
            json.dumps(dbg, ensure_ascii=False, indent=2),
            "",
            [],
        )

    topk_list, top1_letters, crop_gallery = classify_crops(
        L.cls,
        L.cls_letters,
        pil,
        boxes,
        pad=crop_pad,
        topk=topk_k,
    )

    src = build_source_text_mt5(
        pil=pil,
        boxes_xyxy=boxes,
        det_scores=det_scores,
        topk_list=topk_list,
        coord_norm=coord_norm,
        order_mode=order_mode,
        rtl=rtl,
    )

    heb = ""
    eng = ""

    if output_mode == "he":
        heb = mt5_generate(L.mt5_he_tok, L.mt5_he, src)
    elif output_mode == "en_direct":
        eng = mt5_generate(L.mt5_en_tok, L.mt5_en, src)
    else:
        heb = mt5_generate(L.mt5_he_tok, L.mt5_he, src)
        if he2en_kind != "none":
            eng = translate_he_to_en(heb, he2en_kind)
        else:
            eng = mt5_generate(L.mt5_en_tok, L.mt5_en, src)

    fig = make_bbox_figure(
        pil,
        boxes,
        labels=top1_letters,
        det_scores=det_scores,
        topk_list=topk_list,
        height=IMG_HEIGHT,
    )

    dbg = {
        "ts": now_ts(),
        "device": get_device(),
        "detector": L.det_name,
        "classifier": L.cls_name,
        "n_boxes": len(boxes),
        "coord_norm": coord_norm,
        "order_mode": order_mode,
        "rtl": bool(rtl),
        "output_mode": output_mode,
        "he2en_kind": he2en_kind,
    }

    return fig, crop_gallery, heb, eng, json.dumps(dbg, ensure_ascii=False, indent=2), src, topk_list


def run_pipeline_tab(
    pil: Optional[Image.Image],
    conf: float,
    iou: float,
    max_det: int,
    crop_pad: float,
    topk_k: int,
    rtl: bool,
    output_mode: str,
    he2en_kind: str,
    coord_norm: str,
    order_mode: str,
) -> Tuple[
    Optional[Any],
    List[Tuple[Image.Image, str]],
    str,
    str,
    str,
    str,
    List[List[Tuple[str, float]]],
]:
    try:
        with _INFER_LOCK:
            return _run_pipeline_tab_impl(
                pil,
                conf,
                iou,
                max_det,
                crop_pad,
                topk_k,
                rtl,
                output_mode,
                he2en_kind,
                coord_norm,
                order_mode,
            )
    except Exception:
        log.exception("run_pipeline_tab failed")
        return None, [], "ERROR", "ERROR", traceback.format_exc(), "", []


def _run_detector_tab_impl(
    pil: Optional[Image.Image],
    conf: float,
    iou: float,
    max_det: int,
) -> Tuple[Optional[Any], str]:
    if pil is None:
        return None, json.dumps({"error": "No input image"}, ensure_ascii=False, indent=2)

    pil = pil.convert("RGB")
    L = ensure_loaded()

    boxes, scores = yolo_predict(L.det, pil, conf, iou, max_det)
    fig = make_bbox_figure(pil, boxes, det_scores=scores, height=IMG_HEIGHT)

    dbg = {
        "ts": now_ts(),
        "detector": L.det_name,
        "n_boxes": len(boxes),
        "boxes_xyxy": boxes,
        "scores": scores,
    }

    return fig, json.dumps(dbg, ensure_ascii=False, indent=2)


def run_detector_tab(
    pil: Optional[Image.Image],
    conf: float,
    iou: float,
    max_det: int,
) -> Tuple[Optional[Any], str]:
    try:
        with _INFER_LOCK:
            return _run_detector_tab_impl(pil, conf, iou, max_det)
    except Exception:
        log.exception("run_detector_tab failed")
        return None, traceback.format_exc()


def _run_classifier_tab_impl(
    pil: Optional[Image.Image],
    conf: float,
    iou: float,
    max_det: int,
    crop_pad: float,
    topk_k: int,
) -> Tuple[List[Tuple[Image.Image, str]], str]:
    if pil is None:
        return [], json.dumps({"error": "No input image"}, ensure_ascii=False, indent=2)

    pil = pil.convert("RGB")
    L = ensure_loaded()

    boxes, det_scores = yolo_predict(L.det, pil, conf, iou, max_det)

    if not boxes:
        return [], json.dumps(
            {"ts": now_ts(), "n_boxes": 0, "msg": "No boxes detected"},
            ensure_ascii=False,
            indent=2,
        )

    topk_list, top1_letters, crop_gallery = classify_crops(
        L.cls,
        L.cls_letters,
        pil,
        boxes,
        pad=crop_pad,
        topk=topk_k,
    )

    details: Dict[str, Any] = {}
    for i, (row, box, sc) in enumerate(zip(topk_list, boxes, det_scores)):
        details[f"Box_{i + 1:02d}"] = {
            "xyxy": box,
            "det_score": float(sc),
            "topk": row,
        }

    dbg = {
        "ts": now_ts(),
        "classifier": L.cls_name,
        "n_boxes": len(boxes),
        "details": details,
    }

    return crop_gallery, json.dumps(dbg, ensure_ascii=False, indent=2)


def run_classifier_tab(
    pil: Optional[Image.Image],
    conf: float,
    iou: float,
    max_det: int,
    crop_pad: float,
    topk_k: int,
) -> Tuple[List[Tuple[Image.Image, str]], str]:
    try:
        with _INFER_LOCK:
            return _run_classifier_tab_impl(pil, conf, iou, max_det, crop_pad, topk_k)
    except Exception:
        log.exception("run_classifier_tab failed")
        return [], traceback.format_exc()


def show_topk_popup(topk_list: List[List[Tuple[str, float]]], evt: gr.SelectData) -> str:
    idx = getattr(evt, "index", None)

    if idx is None or not topk_list or idx < 0 or idx >= len(topk_list):
        return ""

    rows = topk_list[idx]
    lines = "\n".join([f"{lab:>2}    {p:.4f}" for lab, p in rows])

    return f"""
    <div class="modal-backdrop" onclick="this.parentElement.innerHTML = ''">
      <div class="modal-card">
        <div class="modal-title">Top-K for crop #{idx + 1}</div>
        <pre class="modal-pre">{lines}</pre>
        <div class="modal-hint">Click anywhere outside to close</div>
      </div>
    </div>
    """


def save_feedback(
    heb_pred: str,
    eng_pred: str,
    heb_corr: str,
    eng_corr: str,
    notes: str,
) -> str:
    try:
        rec = {
            "ts": now_ts(),
            "heb_pred": heb_pred,
            "eng_pred": eng_pred,
            "heb_corr": heb_corr,
            "eng_corr": eng_corr,
            "notes": notes,
        }

        os.makedirs(os.path.dirname(FEEDBACK_PATH), exist_ok=True)
        with open(FEEDBACK_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        return f"Saved to {FEEDBACK_PATH}"
    except Exception:
        log.exception("save_feedback failed")
        return traceback.format_exc()


def disable_button(label: str = "Running...") -> Any:
    return gr.update(value=label, interactive=False)


def enable_button(label: str) -> Any:
    return gr.update(value=label, interactive=True)


# -----------------------------------------------------------------------------
# Examples helper
# -----------------------------------------------------------------------------

if EXAMPLES_DIR.exists():
    gr.set_static_paths([str(EXAMPLES_DIR)])


def list_example_images(max_n: int = 24) -> List[List[str]]:
    if not EXAMPLES_DIR.exists():
        return []

    paths = [
        p
        for p in sorted(EXAMPLES_DIR.iterdir())
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    ]

    return [[str(p)] for p in paths[:max_n]]


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------

CSS = """
:root {
  --paleo-primary: #0891b2;
  --paleo-primary-dark: #0e7490;
  --paleo-primary-deep: #0369a1;
  --paleo-accent: #38bdf8;
  --paleo-soft: #ecfeff;
  --paleo-border: #bae6fd;
  --paleo-text: #0f172a;
}

.gradio-container {
  color: var(--paleo-text) !important;
}

button.primary,
.gradio-container button.primary,
.gr-button-primary {
  background: linear-gradient(90deg, #0891b2, #0284c7) !important;
  border-color: #0e7490 !important;
  color: white !important;
  box-shadow: 0 8px 22px rgba(8, 145, 178, 0.18) !important;
}

button.primary:hover,
.gradio-container button.primary:hover,
.gr-button-primary:hover {
  background: linear-gradient(90deg, #0e7490, #0369a1) !important;
  border-color: #075985 !important;
}

button.primary:disabled,
.gradio-container button.primary:disabled,
.gr-button-primary:disabled {
  background: linear-gradient(90deg, #67e8f9, #7dd3fc) !important;
  border-color: #67e8f9 !important;
  color: #0f172a !important;
  opacity: 0.75 !important;
}

.tab-nav button.selected,
.tab-nav button[aria-selected="true"],
.gradio-container [role="tab"][aria-selected="true"] {
  color: #0369a1 !important;
  border-color: #0891b2 !important;
}

input[type="range"] {
  accent-color: #0891b2 !important;
}

textarea:focus,
input:focus,
select:focus {
  border-color: #0891b2 !important;
  box-shadow: 0 0 0 2px rgba(8, 145, 178, 0.18) !important;
}

.gradio-container .block,
.gradio-container .panel {
  border-color: rgba(186, 230, 253, 0.9) !important;
}

.gradio-container .label-wrap,
.gradio-container .accordion {
  border-color: rgba(186, 230, 253, 0.9) !important;
}

#topk_modal .modal-backdrop {
  position: fixed;
  inset: 0;
  background: rgba(15, 23, 42, 0.35);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 9999;
}
#topk_modal .modal-card {
  background: white;
  border: 1px solid #bae6fd;
  border-radius: 14px;
  padding: 18px 22px;
  min-width: 280px;
  max-width: 520px;
  box-shadow: 0 10px 40px rgba(8, 145, 178, 0.25);
}
#topk_modal .modal-title {
  color: #0369a1;
  font-size: 18px;
  font-weight: 700;
  margin-bottom: 10px;
}
#topk_modal .modal-pre {
  font-size: 16px;
  line-height: 1.25;
  margin: 0;
  white-space: pre;
}
#topk_modal .modal-hint {
  opacity: 0.6;
  margin-top: 10px;
}
"""


def build_theme() -> Any:
    try:
        return gr.themes.Soft(
            primary_hue="cyan",
            secondary_hue="sky",
            neutral_hue="slate",
        )
    except Exception:
        return None


_BLOCKS_KWARGS: Dict[str, Any] = {
    "title": "Paleo-Hebrew Tablet Reader",
}

_theme = build_theme()
if _theme is not None:
    _BLOCKS_KWARGS["theme"] = _theme

# Gradio 6 moved css from Blocks() to launch(). Older versions may still expect css in Blocks().
if not supports_param(gr.Blocks.launch, "css") and supports_param(gr.Blocks, "css"):
    _BLOCKS_KWARGS["css"] = CSS


with gr.Blocks(**_BLOCKS_KWARGS) as demo:
    gr.Markdown("# Paleo-Hebrew Epigraphy Pipeline")

    topk_state = gr.State([])
    modal = gr.HTML(value="", elem_id="topk_modal")

    with gr.Row():
        with gr.Column(scale=3):
            inp = gr.Image(type="pil", label="Input Image", height=IMG_HEIGHT)

            with gr.Accordion("Vision Settings", open=True):
                conf = gr.Slider(0.05, 0.95, value=0.25, step=0.01, label="Box Confidence")
                iou = gr.Slider(0.10, 0.90, value=0.45, step=0.01, label="IoU")
                max_det = gr.Slider(1, 200, value=80, step=1, label="Max Detections")
                crop_pad = gr.Slider(0.0, 0.8, value=0.20, step=0.01, label="Crop Pad")
                topk_k = gr.Slider(1, 10, value=5, step=1, label="Classifier Top-K")

            with gr.Accordion("mT5 + Translation Settings", open=True):
                rtl = gr.Checkbox(
                    value=True,
                    label="RTL order (right to left) [for reading-order mode]",
                )
                order_mode = gr.Radio(["reading", "detector"], value="reading", label="Box order")
                coord_norm = gr.Radio(
                    ["boxes", "det", "none"],
                    value="boxes",
                    label="coord_norm (like training)",
                )
                output_mode = gr.Radio(
                    ["he", "en_direct", "he_then_en"],
                    value="he_then_en",
                    label="Output mode",
                )
                he2en_kind = gr.Dropdown(
                    choices=TRANSLATOR_CHOICES,
                    value="opus",
                    label="Hebrew to English translator",
                )

            with gr.Accordion("How to run a VLM yourself (optional)", open=False):
                gr.Markdown(
                    """
If you want a VLM post-OCR step (outside this Space), run it locally / on a GPU box.

High-level steps:

1. Build a `tools_text` block from YOLO+classifier: `[DETECTOR ...]` + `[CLASSIFIER topk]`.
2. Feed image + prompt + tools_text into your Qwen3-VL LoRA model.

Pseudo-code:

    from transformers import AutoProcessor, AutoModelForImageTextToText

    proc = AutoProcessor.from_pretrained("mr3vial/paleo-hebrew-qwen3-vl-lora-post-ocr-processing")
    model = AutoModelForImageTextToText.from_pretrained("mr3vial/paleo-hebrew-qwen3-vl-lora-post-ocr-processing")

    prompt = "Transcribe the text on the tablet.\n\n" + tools_text
    inputs = proc(text=prompt, images=[pil_image], return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=64)
    pred = proc.batch_decode(out, skip_special_tokens=True)[0]
"""
                )

            ex = list_example_images()
            if ex:
                gr.Markdown("### Examples")
                gr.Examples(examples=ex, inputs=[inp], cache_examples=False)

        with gr.Column(scale=7):
            with gr.Tabs():
                with gr.Tab("End-to-End Pipeline"):
                    run_pipe_btn = gr.Button("Run Full Pipeline", variant="primary")
                    out_fig_pipe = gr.Plot(label="Detections (hover tooltip)")

                    with gr.Row():
                        out_he_pipe = gr.Textbox(label="Hebrew", lines=2, interactive=False)
                        out_en_pipe = gr.Textbox(label="English", lines=2, interactive=False)

                    out_crops_pipe = gr.Gallery(columns=8, height=360, label="Letter crops")

                    with gr.Accordion("Debug", open=False):
                        out_dbg_pipe = gr.Code(label="Debug JSON", language="json")
                        out_mt5_src = gr.Textbox(
                            label="mT5 source text (fed into mT5)",
                            lines=10,
                            interactive=False,
                        )

                with gr.Tab("Detector"):
                    run_det_btn = gr.Button("Run Detector", variant="primary")
                    out_fig_det = gr.Plot(label="Detected Bounding Boxes")
                    out_json_det = gr.Code(label="Raw Detection Output", language="json")

                with gr.Tab("Classifier"):
                    run_cls_btn = gr.Button("Run Classifier", variant="primary")
                    out_crops_cls = gr.Gallery(columns=8, height=360, label="Isolated Letter Crops")
                    out_json_cls = gr.Code(label="Top-K per Box", language="json")

                with gr.Tab("Feedback"):
                    gr.Markdown("Submit corrections to improve the dataset.")
                    heb_corr = gr.Textbox(label="Correct Hebrew", lines=2)
                    eng_corr = gr.Textbox(label="Correct English", lines=2)
                    notes = gr.Textbox(label="Notes", lines=2)
                    save_btn = gr.Button("Submit Feedback")
                    save_status = gr.Textbox(label="Status", interactive=False)

    # -------------------------------------------------------------------------
    # Wiring
    # -------------------------------------------------------------------------

    heavy_event_kwargs = kwargs_supported_by(
        run_pipe_btn.click,
        {
            "api_name": False,
            "concurrency_limit": 1,
            "concurrency_id": MODEL_CONCURRENCY_ID,
        },
    )

    disable_click_kwargs = kwargs_supported_by(
        run_pipe_btn.click,
        {
            "queue": False,
            "api_name": False,
        },
    )

    # End-to-End Pipeline.
    ev_pipe = run_pipe_btn.click(
        fn=lambda: disable_button("Running..."),
        inputs=[],
        outputs=[run_pipe_btn],
        **disable_click_kwargs,
    )
    ev_pipe = ev_pipe.then(
        fn=run_pipeline_tab,
        inputs=[
            inp,
            conf,
            iou,
            max_det,
            crop_pad,
            topk_k,
            rtl,
            output_mode,
            he2en_kind,
            coord_norm,
            order_mode,
        ],
        outputs=[
            out_fig_pipe,
            out_crops_pipe,
            out_he_pipe,
            out_en_pipe,
            out_dbg_pipe,
            out_mt5_src,
            topk_state,
        ],
        **heavy_event_kwargs,
    )
    ev_pipe.then(
        fn=lambda: enable_button("Run Full Pipeline"),
        inputs=[],
        outputs=[run_pipe_btn],
        **disable_click_kwargs,
    )

    out_crops_pipe.select(
        fn=show_topk_popup,
        inputs=[topk_state],
        outputs=[modal],
        **kwargs_supported_by(out_crops_pipe.select, {"api_name": False}),
    )

    # Detector tab.
    ev_det = run_det_btn.click(
        fn=lambda: disable_button("Running..."),
        inputs=[],
        outputs=[run_det_btn],
        **disable_click_kwargs,
    )
    ev_det = ev_det.then(
        fn=run_detector_tab,
        inputs=[inp, conf, iou, max_det],
        outputs=[out_fig_det, out_json_det],
        **heavy_event_kwargs,
    )
    ev_det.then(
        fn=lambda: enable_button("Run Detector"),
        inputs=[],
        outputs=[run_det_btn],
        **disable_click_kwargs,
    )

    # Classifier tab.
    ev_cls = run_cls_btn.click(
        fn=lambda: disable_button("Running..."),
        inputs=[],
        outputs=[run_cls_btn],
        **disable_click_kwargs,
    )
    ev_cls = ev_cls.then(
        fn=run_classifier_tab,
        inputs=[inp, conf, iou, max_det, crop_pad, topk_k],
        outputs=[out_crops_cls, out_json_cls],
        **heavy_event_kwargs,
    )
    ev_cls.then(
        fn=lambda: enable_button("Run Classifier"),
        inputs=[],
        outputs=[run_cls_btn],
        **disable_click_kwargs,
    )

    save_btn.click(
        fn=save_feedback,
        inputs=[out_he_pipe, out_en_pipe, heb_corr, eng_corr, notes],
        outputs=[save_status],
        **kwargs_supported_by(save_btn.click, {"api_name": False}),
    )


def launch_app() -> None:
    queue_kwargs: Dict[str, Any] = {
        "max_size": QUEUE_MAX_SIZE,
    }

    if supports_param(demo.queue, "default_concurrency_limit"):
        queue_kwargs["default_concurrency_limit"] = DEFAULT_CONCURRENCY_LIMIT

    app = demo.queue(**queue_kwargs)

    launch_kwargs: Dict[str, Any] = {
        "show_error": True,
    }

    if supports_param(demo.launch, "ssr_mode"):
        launch_kwargs["ssr_mode"] = False

    if supports_param(demo.launch, "css"):
        launch_kwargs["css"] = CSS

    log.info("Launching Gradio app with queue_kwargs=%s launch_kwargs=%s", queue_kwargs, launch_kwargs)
    app.launch(**launch_kwargs)


if __name__ == "__main__":
    launch_app()
