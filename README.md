# Rootstalk — Crop Disease Decision Support System

## Project Overview

Rootstalk is a mobile-based decision support system that helps farmers identify crop diseases by comparing field photos against a reference image gallery. The farmer selects a crop part, captures a photo, and the system returns a similarity percentage (e.g., "87% match to Leaf Rust") along with a confidence-backed recommendation — **Take Action**, **Monitor**, or **Low Concern**.

This is not an automatic diagnosis engine. It is a tool that supports the farmer's own judgment.

---

## What Has Been Built — Week 1 & Week 2

### Week 1 — Similarity PoC (Proof of Concept)

**Goal:** Validate that image similarity can distinguish between crop disease symptoms before investing in full model training.

**Approach:** Used DINOv2 (ViT-S/14), a state-of-the-art vision transformer from Meta AI, to convert images into 384-dimensional embedding vectors. Cosine similarity between vectors produces a percentage match score.

**Dataset:** 1,639 images across 10 symptom classes:

| Symptom | Images |
|---|---|
| Leaf color change | 468 |
| Leaf chewed portion | 414 |
| Leaf shape change | 310 |
| Leaf Healthy | 200 |
| Fruit bored holes | 93 |
| Stem bored holes | 58 |
| Burnt appearance | 46 |
| Fruit cracking | 27 |
| Galls | 13 |
| Stem cracking | 12 |
| **Total** | **1,639** |

**PoC Validation Results:**

| Test Image | Top Match | Similarity |
|---|---|---|
| Leaf chewed portion | Leaf chewed portion | 100% ✓ |
| Root galls (Google image) | Galls | 73.4% ✓ |
| Stem cracking | Stem cracking | 63.7% ✓ |

**Key finding:** Stem cracking scored lower (63.7%) due to having only 12 reference images — confirmed data imbalance as a risk.

**Project Structure:**
```
image_processing/
├── src/model/         # DINOv2 encoder, similarity, fine-tuning
├── src/data/          # Dataset, gallery builder, preprocessing
├── src/api/           # FastAPI server, routes, schemas
├── scripts/           # Build gallery, train, evaluate, export
├── notebooks/         # PoC demo
├── data/raw/          # 1,639 images across 10 symptom folders
├── data/embeddings/   # Pre-computed embeddings (.npy + labels.json)
└── config/            # Config YAML and gallery metadata
```

---

### Week 2 — Fine-Tuning & Accuracy Optimization

**Goal:** Domain-specialize DINOv2 embeddings so the model understands plant disease symptoms, not just generic visual features.

**Why Fine-Tune?**
Out-of-the-box DINOv2 understands general visual features. But crop disease symptoms have subtle differences that a generic model may miss. Fine-tuning teaches the model which visual differences matter most for crop diseases.

**Method — Triplet Loss Training:**
- **Anchor:** An image of symptom A (e.g., a gall)
- **Positive:** A different image of the same symptom A
- **Negative:** An image of any other symptom
- The loss function pushes same-symptom embeddings closer and different-symptom embeddings further apart in the 128-dimensional projection space.

**Training Setup:**
- Model: DINOv2 ViT-S/14 + custom projection head (384 → 256 → 128 dimensions)
- Loss: TripletMarginLoss (margin = 0.3)
- Hardware: Google Colab T4 GPU
- Total epochs: 10 (Phase 1 + Phase 2)
- Batch size: 16

**Two-Phase Training:**

| Phase | Epochs | What Happens | Final Loss |
|---|---|---|---|
| Phase 1 | 1–5 | Backbone frozen, only projection head trains | 0.0182 |
| Phase 2 | 6–10 | Full backbone unfrozen, entire model fine-tunes | 0.0058 |

**Loss progression across all 10 epochs:**
```
Phase 1: 0.0780 → 0.0430 → 0.0312 → 0.0257 → 0.0182
Phase 2: 0.0215 → 0.0182 → 0.0144 → 0.0077 → 0.0058
```
The slight rise at Phase 2 Epoch 1 is expected — when the backbone unfreezes, weights shift temporarily before converging to a better minimum.

---

## Evaluation Results

**Metric:** Precision@5 and Recall@5 — "When I query with a symptom image, are the top 5 results also the same symptom?"

### Per-Symptom Results

| Symptom | Baseline P@5 | Fine-tuned P@5 | Baseline R@5 | Fine-tuned R@5 |
|---|---|---|---|---|
| Leaf chewed portion | 0.96 | 1.00 ✓ | 0.01 | 0.01 |
| Fruit bored holes | 0.96 | 1.00 ✓ | 0.05 | 0.05 |
| Leaf color change | 0.80 | 0.84 ↑ | 0.01 | 0.01 |
| Stem bored holes | 0.80 | 1.00 ✓ | 0.07 | 0.09 |
| Leaf shape change | 0.68 | 0.60 ↓ | 0.01 | 0.01 |
| Fruit cracking | 0.76 | 1.00 ✓ | 0.14 | 0.19 |
| Galls | 0.64 | 1.00 ✓ | 0.25 | 0.38 |
| Leaf Healthy | 0.64 | 0.88 ↑ | 0.02 | 0.02 |
| Burnt appearance | 0.44 | 0.72 ↑ | 0.05 | 0.08 |
| Stem cracking | 0.44 | 1.00 ✓ | 0.18 | 0.42 |

### Overall Results

| Metric | Baseline DINOv2 | Fine-tuned Model | Improvement |
|---|---|---|---|
| Precision@5 | 0.7120 | 0.9040 | **+27.0%** |
| Recall@5 | 0.0786 | 0.1257 | **+60.0%** |

---

## Similarity Thresholds

Based on the fine-tuned model's performance, thresholds were calibrated:

| Similarity Score | Recommendation | Meaning |
|---|---|---|
| ≥ 80% | **Take Action** | High confidence match — consult expert immediately |
| 60–79% | **Monitor** | Moderate confidence — observe and recheck |
| < 60% | **Low Concern** | Low confidence — likely not a match |

---

## Known Gaps & Next Steps

| Gap | Detail | Solution |
|---|---|---|
| Data imbalance | Stem cracking (12 images), Galls (13 images) | Collect more field images |
| Leaf shape change dropped | Fine-tuned P@5 = 0.60 vs baseline 0.68 | More diverse training images needed |
| Recall is low overall | Top 5 out of 1,639 images | Increase k or expand gallery |
| mAP not yet computed | Only Precision@K and Recall@K done | Add mAP in next evaluation cycle |
| Dataset expansion | PlantVillage has 54,000+ images | Integrate in upcoming weeks |

---

## Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| Embedding model | DINOv2 ViT-S/14 | Convert images to 384-dim vectors |
| Fine-tuning | PyTorch + TripletMarginLoss | Domain specialization |
| Similarity search | Cosine similarity + FAISS | Fast nearest-neighbor lookup |
| API | FastAPI | Serve similarity scores |
| Preprocessing | OpenCV + torchvision | Image normalization |
| Version control | GitHub | Code management |
| Training infrastructure | Google Colab T4 GPU | Model training |

---

## Repository

**GitHub:** https://github.com/deekshad13/image_processing

---

*Last updated: Week 2 complete — Fine-tuning & Evaluation done.*
