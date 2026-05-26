# Paleo-Hebrew Epigraphy Pipeline

Interactive Gradio demo for detecting Paleo-Hebrew letter boxes, classifying glyphs, and running mT5 post-OCR / translation.

## Run locally

Install dependencies from `requirements.txt`, then start the app from the repository root:

```bash
python app.py
```

### 1. Prerequisites

- **Python 3.10+** (recommended)
- **Git** and **Git LFS** (example images under `examples/` may be stored with LFS)
- **Hugging Face Hub access** — model weights are downloaded on first run (public repos by default; a token is only needed for private or gated assets)
- **GPU strongly recommended** — PyTorch uses CUDA when available; CPU works but first inference and optional translators are slow

### 2. Clone and install dependencies

```bash
git clone <repository-url>
cd <repository-directory>

# If examples/ are LFS pointers, pull the real files:
git lfs pull
```

Create a virtual environment (recommended), then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -U pip
pip install "gradio==6.6.0"
pip install -r requirements.txt
```

If `requirements.txt` is missing in your checkout, create it with:

```text
gradio>=4.0.0
huggingface_hub>=0.20.0
transformers>=4.40.0
sentencepiece
torch
numpy
Pillow

ultralytics>=8.0.0
onnxruntime

timm
safetensors
plotly
onnx
peft
qwen-vl-utils
```

For **CUDA**, install a PyTorch build for your system from [pytorch.org](https://pytorch.org/get-started/locally/), e.g.:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

### 3. Start the app

From the repo root (where `app.py` and `examples/` live):

```bash
python app.py
```

Gradio prints a local URL (typically `http://127.0.0.1:7860`). Open it in a browser.

**First request** triggers lazy loading of detector, classifier, and mT5 weights from the Hugging Face Hub. Repository IDs and filenames are set via environment variables (defaults in `app.py`). Expect a long cold start and several GB of downloads.

Optional: authenticate for higher Hub rate limits or private assets:

```bash
export HF_TOKEN=hf_...   # or: huggingface-cli login
python app.py
```

### 4. Runtime defaults


| Setting       | Default                       | Notes                              |
| ------------- | ----------------------------- | ---------------------------------- |
| Entry command | `python app.py`               | Same as typical Gradio deployments |
| Gradio        | 6.6.0                         | `pip install "gradio==6.6.0"`      |
| SSR           | off                           | Unless `PALEO_ENABLE_GRADIO_SSR=1` |
| Examples      | `examples/` next to `app.py`  | Gallery images                     |
| Queue         | `max_size=4`, concurrency `1` | Overridable via env (below)        |


## Project layout

```text
.
├── app.py              # Gradio UI + pipeline (run this)
├── examples/           # Sample images for the Examples gallery
├── README.md
└── requirements.txt    # Python dependencies
```

## Environment variables (optional)

Defaults are in `app.py`. Override to use different Hub repositories or tune behavior:


| Variable                    | Purpose                                 |
| --------------------------- | --------------------------------------- |
| `DET_REPO_ID`               | YOLO detector Hub repository            |
| `DET_FILENAME`              | Detector weights filename               |
| `DET_FALLBACK_PT`           | PT fallback weights filename            |
| `CLS_REPO_ID`               | Classifier Hub repository               |
| `CLS_WEIGHTS`               | Classifier checkpoint filename          |
| `CLS_CLASSES`               | Class labels JSON filename              |
| `CLS_MODEL_NAME`            | timm model architecture name            |
| `MT5_BOX2HE_REPO`           | Box → Hebrew mT5 repository             |
| `MT5_BOX2EN_REPO`           | Box → English mT5 repository            |
| `PALEO_ENABLE_GRADIO_SSR`   | Set to `1` to enable Gradio SSR         |
| `QUEUE_MAX_SIZE`            | Gradio queue size (default `4`)         |
| `DEFAULT_CONCURRENCY_LIMIT` | Max concurrent heavy jobs (default `1`) |
| `FEEDBACK_PATH`             | Feedback tab JSONL output path          |


Example:

```bash
export DET_REPO_ID=your-org/your-detector-repo
export QUEUE_MAX_SIZE=2
python app.py
```

## Troubleshooting

- **Missing `examples/` images** — run `git lfs pull` after clone.
- `**ModuleNotFoundError`** — ensure the venv is active and `requirements.txt` is installed; lazy imports need `ultralytics`, `timm`, and `transformers` before the first Run click.
- **Out of memory** — use a GPU with enough VRAM; avoid optional Hebrew→English translators (NLLB / M2M100) unless needed; use `he` or `en_direct` output modes on small machines.
- **Slow or timing out on CPU** — normal for the full pipeline; use a GPU or allow time for the first download and model load.

## License

MIT.
