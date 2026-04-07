# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Rootstalk is a **crop disease image intelligence engine** for the EywaFarm/Roots Talk platform. It provides single-symptom verification: a farmer on a symptom page uploads a photo, and the system tells them how closely it matches THAT specific symptom.

Long-term goal: every image processed makes the engine stronger — building a production-grade agricultural image intelligence asset.

## Development Commands

```bash
# Setup (Python 3.13 arm64 venv)
source venv/bin/activate
pip install -r Requirements.txt

# Build gallery (must run before starting API)
python scripts/build_gallery.py

# Run API server
uvicorn src.api.server:app --host 0.0.0.0 --port 8000

# Run tests
pytest tests/test_api.py -v

# Train fine-tuned model
python scripts/train.py

# Evaluate verification accuracy
python scripts/evaluate.py

# PoC demo
python notebooks/02_poc_demo.py galls data/raw/Galls/some_image.jpg
```

## Architecture

### Core Flow

```
POST /verify/{symptom_id} + image
  → Validate image (format, blur, brightness — thresholds configurable via API)
  → Embed via active model engine
  → Cosine similarity ONLY against that symptom's gallery
  → Mean of top-N similarities → single percentage
  → Response: similarity_pct + confidence + recommendation
```

### Multi-Engine System

`src/model/engines.py` provides a pluggable model registry. Six engines available:

| Engine ID | Model | Family | Params | Embed Dim |
|-----------|-------|--------|--------|-----------|
| `dinov2_vits14` | DINOv2 ViT-S/14 | DINOv2 | 21M | 384 |
| `dinov2_vitb14` | DINOv2 ViT-B/14 | DINOv2 | 86M | 768 |
| `clip_vitb32` | CLIP ViT-B/32 | CLIP | 151M | 512 |
| `clip_vitl14` | CLIP ViT-L/14 | CLIP | 428M | 768 |
| `convnext_base` | ConvNeXt Base | CNN | 89M | 1024 |
| `resnet50` | ResNet-50 | CNN | 25M | 2048 |

`POST /compare-engines/{symptom_id}` runs all engines against the same image and returns ranked results with timing.

### API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Health check |
| GET | `/gallery` | List all symptoms |
| GET | `/gallery/{symptom_id}` | Symptom detail |
| GET | `/gallery/{symptom_id}/images` | Reference image URLs |
| POST | `/verify/{symptom_id}` | Single-engine verification |
| POST | `/compare-engines/{symptom_id}` | Multi-engine comparison |
| GET | `/engines` | List available engines |
| GET | `/settings` | Get validation/similarity config |
| PUT | `/settings` | Update config at runtime |
| GET | `/images/{path}` | Static file serving for reference images |

### Key Modules

- **`src/config.py`** — YAML config loader, cached globally
- **`src/data/symptom_registry.py`** — symptom_id ↔ folder name mapping (source of truth)
- **`src/data/preprocessing.py`** — Centralized image transform. Never duplicate.
- **`src/data/gallery_builder.py`** — Numpy-based cosine similarity search per symptom
- **`src/model/engines.py`** — Multi-engine registry with lazy loading and caching
- **`src/model/encoder.py`** — Default model loader (fine-tuned or baseline DINOv2)
- **`src/model/fine_tune.py`** — CropSimilarityModel + training pipeline
- **`src/api/routes.py`** — All API endpoints. Model + gallery loaded at import time.
- **`test_ui.html`** — Full test UI with symptom selection, upload, reference images, engine comparison, and settings panel

### Configuration

All in `config/config.yaml` — also adjustable at runtime via `PUT /settings`:
- Symptom registry (symptom_id → folder, display_name)
- Similarity thresholds (high=80%, medium=60%)
- Validation params (blur=15, brightness 30-240, file size 100MB)
- Transform params, training params, model paths

### Data

- `data/raw/{symptom}/` — 1,641 source images across 10 symptom classes
- `data/embeddings/{symptom}/embeddings.npy` — Pre-computed embeddings
- `data/embeddings/fine_tuned_model.pt` — Fine-tuned weights (not on disk yet — trained on Colab)

### Important Patterns

- **Module-level model loading**: routes.py loads model+gallery at import time
- **Engine lazy loading**: engines.py loads models on first use, caches them
- **CORS enabled**: server.py allows all origins for dev UI
- **Static image serving**: `/images/` serves from `data/raw/` for reference thumbnails
- **Scripts run from repo root**: use `sys.path.insert` for `src.*` imports
- **faiss-cpu conflicts with torch on macOS arm64**: gallery_builder.py uses numpy instead
