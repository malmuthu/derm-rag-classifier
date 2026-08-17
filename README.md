# Dermatology Lesion Classifier + Retrieval-Augmented Explanation

A skin lesion classifier (ResNet50 / ViT-B/16, trained on HAM10000) combined with a
retrieval layer that surfaces visually similar historical cases for any prediction —
built to move beyond a bare classification label toward a grounded, evidence-backed
explanation. All evaluation here is done via macro-averaged metrics and per-class breakdowns, computed on a *lesion-level* held-out test set with
   leakage explicitly checked and tested for

---

## Repository Structure

```
derm-rag-classifier/
├── configs/
├── data/                  # gitignored - raw images, processed splits, case DB
├── notebooks/
├── src/
│   ├── data/              # Dataset/DataLoader
│   ├── models/            # model factory, losses, training loop, evaluation
│   ├── retrieval/         # embeddings, FAISS index, similarity search
│   ├── utils/             # shared config loading
├── tests/                 # pytest - data integrity + retrieval quality         
└── checkpoints/           # gitignored
```

---

## Setup

```bash
conda create -n derm-rag python=3.11
conda activate derm-rag
pip install pandas numpy matplotlib pillow scikit-learn pyyaml torch torchvision \
            faiss-cpu tqdm pytest jupyter ipykernel
```

Download HAM10000 and place it at:
```
data/raw/HAM10000_metadata.csv
data/raw/HAM10000_images_part_1/
data/raw/HAM10000_images_part_2/
```

```bash
python src/data/make_splits.py          # lesion-level train/val/test split
python src/models/train.py --config configs/resnet50_dropout.yaml
python src/models/train.py --config configs/vit_b_16_dropout.yaml
python src/retrieval/build_case_db.py   # embed training images
python src/retrieval/build_index.py     # build FAISS index
```

Run tests:
```bash
pytest tests/ -v
```

---

## 1 — Data: HAM10000

Dermoscopic images across 7 diagnostic classes. Full EDA in
`notebooks/01_eda_ham10000.ipynb`. Key findings that shaped every downstream decision:

| Finding | Implication |
|---|---|
| Severe class imbalance (~67% `nv`, ~1.1% `df`) | Accuracy is a misleading metric; macro F1 + per-class recall used instead; class-weighted loss applied |
| ~26% of lesions have multiple images | Splits done at the lesion level, not image level |
| `mel` vs. `nv` are visually hard to distinguish | Motivates the retrieval/explanation layers |
| Body site and age showed no strong class-discriminating pattern | Image-only modeling approach rather than multi-modal fusion |

**Split:** 70/15/15, stratified by class, split at the `lesion_id` level. Verified
programmatically (`tests/test_splits.py`) — zero lesion overlap between splits, all
classes present in every split, class proportions preserved within 5 percentage points.

---

## 2 — Classifier

### Architecture

Two backbones trained and compared under identical conditions (same splits, same
augmentation, same optimizer schedule) to isolate the effect of architecture choice:

- **ResNet50**, ImageNet-pretrained (`IMAGENET1K_V2` weights)
- **ViT-B/16**, ImageNet-pretrained (`IMAGENET1K_V1` weights)

### Transfer learning strategy: two-phase progressive unfreezing

- **Phase 1:** entire pretrained backbone frozen, only the new classification head trained.
- **Phase 2:** later layers unfrozen for fine-tuning at a 10x lower learning rate
  (ResNet: `layer3`+`layer4`+`fc`; ViT: last 4 of 12 encoder blocks + head).

Chosen over full fine-tuning to reduce the risk of catastrophic forgetting on a
dataset of this size. Note ResNet's partial split and ViT's are not equivalently conservative in practice,
despite both being "partial fine-tuning" by layer count. Documented rather than forced to match.

### Class imbalance handling

Class-weighted cross-entropy loss (inverse-frequency weighting, computed from the
training set only). 

### Checkpoint selection

Best checkpoint saved by **validation macro F1**, with per-class `mel` recall tracked
alongside as a secondary, clinically-motivated metric.

### Experiment log

Rather than accept the first working configuration, the following changes were tested
in isolation against a fixed baseline (ResNet50, class-weighted loss, original
unfreeze split, no dropout) to understand what actually helped:

| Change tested | Result |
|---|---|
| Dropout (p=0.3) before final classification layer | Macro F1 improved, gains distributed across nearly all classes, no class regressed significantly |
| `mel`-specific decision threshold override (predict `mel` if P(mel) > threshold, regardless of argmax) | At threshold=0.3: **both** `mel` precision and recall improved simultaneously over plain argmax — not just a trade-off | 


### Final results (test set, lesion-level held-out)

| Model | Macro F1 | `mel` recall | `mel` precision |
|---|---|---|---|
| ResNet50 (dropout, threshold=0.3) | 0.71 | 0.71 | 0.56 |
| **ViT-B/16 (dropout, threshold=0.3)** | **0.79** | 0.72 | 0.44 |

ViT-B/16 outperformed ResNet50 on the overall macro F1 and on `mel` recall, despite
having a smaller fraction of its parameters unfrozen during fine-tuning.

**Limitation:** further tuning attempts beyond dropout consistently produced
either no improvement or regressions concentrated on rare/important classes — this
result is treated as a reasonable stopping point for this stage.

---

## 3 — Retrieval layer

### Design

- **Embeddings:** the 768-dimensional feature vector immediately preceding ViT-B/16's
  final classification layer. ViT was chosen as the embedding source since it
  was the stronger-performing classifier.
- **Vector store:** FAISS, `IndexFlatIP`. Embeddings are L2-normalized so inner product is equivalent to cosine
  similarity.
- **Case database:** built from the **training set only**. Documented limitation: since
  ViT was trained on these exact images, retrieved "historical cases" are images the
  model has already seen during training.

### Retrieval quality — validated, not assumed

Queried using held-out **test-set** images (never seen by the embedding model or the
case database), one query per class:

| Result | Finding |
|---|---|
| Top-1 diagnosis match | Correct for all 7 classes tested |
| Top-3 diagnosis match | Correct for all 7 classes tested |
| Top-5 diagnosis match (≥3 of 5 same diagnosis) | 6 of 7 classes — only `df` (rarest class, ~115 lesions) fell short, plausibly due to limited case-database coverage |

Notably, retrieval quality on `mel` — the classifier's weakest class — was strong,
suggesting retrieval can surface useful, mostly-correct context even where the
classifier's own single-label prediction is uncertain. This is the core empirical
justification for building a retrieval/explanation layer on top of an imperfect
classifier.

Regression-tested in `tests/test_retrieval.py`: case DB/index row-alignment integrity,
self-retrieval sanity check, and the per-class diagnosis-match-rate check above (asserted ≥70% of classes
must show ≥3/5 top-5 match).

---

## Current status and next steps

**In progress:**
- **Stage 4 — explanation layer:** generating grounded text explanations from
  retrieved cases (e.g., "resembles N similar cases, primarily diagnosed as X") without
  hallucinating beyond what retrieval actually returned.
- **Stage 5 — deployment:** an interactive demo (Streamlit) — upload an image, see
  prediction + retrieved cases + explanation together.
- **Stage 6 — polish:** finalize this document, add a model card.

---

## Limitations

- HAM10000 is not demographically representative — performance on darker skin tones
  is untested and likely worse; this is a real, unaddressed limitation of the current
  model.
- `dx_type` confirmation method varies across the dataset (histopathology vs.
  follow-up vs. consensus) — the model does not distinguish confidence level by
  ground-truth source.
- This is a portfolio/research project, not a validated clinical tool, and is not
  intended for diagnostic use.
