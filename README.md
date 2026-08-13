# DermAlert Framework

DermAlert is an end-to-end computer vision framework for smartphone-acquired
dermatological imagery. It pre-processes a lesion photograph (contrast
enhancement and hair removal), then classifies it as benign or
malignant/inflamed with an ensemble of fine-tuned convolutional networks.

🔗 **Live Deployment URL:** [https://dermalert-analytics.onrender.com](https://dermalert-analytics.onrender.com)

> **Not a medical device.** This is a student research project. It is a
> screening aid, not a diagnosis. It misses roughly one malignant lesion in
> eight, and it was trained on dermoscopic images, so ordinary phone photos are
> outside its training distribution. Any lesion that changes in size, shape or
> colour should be seen by a dermatologist regardless of what this outputs.

---

## Key Features

* **Shared pre-processing pipeline:** adaptive contrast enhancement (CLAHE) and
  Dull-Razor digital hair removal, defined once in `features.py` so training and
  serving cannot drift apart.
* **Ensemble inference engine:** five independently trained EfficientNet-B0
  models, each evaluated with 4-view flip test-time augmentation, averaged.
* **Lightweight deployment:** models are exported to ONNX and served with
  onnxruntime, so the deployed app never imports torch.
* **Responsive interface:** shows each pre-processing stage alongside the result.

---

## Results

Benign vs malignant on HAM10000. One image per lesion (HAM10000 ships 10015
images of only 7470 distinct lesions; keeping the duplicates lets the same
lesion appear on both sides of a split and reports memorisation as accuracy).
The test set is 15% of the class-balanced 2338, held out at seed 42, and is the
same split every number below was measured on. The epoch is chosen on a
separate 400-image validation set, and the test set is scored once.

| Metric | Score |
|---|---|
| **Held-out test accuracy** | **89.46%** (95% CI 85.81-92.26%) |
| Balanced accuracy | 89.45% |
| Recall, malignant | 87.43% |
| ROC AUC | 0.9608 |

Read that interval before quoting the headline. The test set is 351 images, so
one image is worth 0.28 points and the honest uncertainty is about +/-3 points.
Differences of a point or two between any two models here are noise, not
progress; only ROC AUC and the much larger out-of-fold estimates below separate
them reliably.

Individual ensemble members score 84.90% to 88.03%; averaging them is worth
roughly 1.5 to 4 points because their errors are only partly correlated.

### How this got from 82% to 89%

The original pipeline was 26 handcrafted features (colour statistics, GLCM, LBP,
contour metrics) into an RBF SVM. It was not under-tuned, it was
feature-limited. Every row below is measured on the split described above:

| Approach | Test accuracy |
|---|---|
| 26 handcrafted features + RBF SVM (original) | 82.91% |
| 593 handcrafted features (448px, tiled multi-angle GLCM, HSV/Lab histograms) + RBF SVM | 84.62% |
| Frozen ResNet50 + EfficientNet-B0 embeddings + RBF SVM | 83.48% |
| Fine-tuned EfficientNet-B3 @300px | 84.62% |
| Fine-tuned EfficientNet-B0 @224px, single model | 84.90% - 88.03% |
| **Ensemble of 5 EfficientNet-B0 seeds** | **89.46%** |

Things that were tried and did **not** help:

* **More data from ISIC-2019.** Adding 6461 extra deduplicated lesions (2403 of
  them malignant, quadrupling the malignant pool) *lowered* validation accuracy
  from 87.0% to 81.8%. ISIC-2019 spans different devices and sites, and the
  domain shift costs more than the extra data gains. The same thing happened to
  the SVM (84.6% to 82.3%).
* **A bigger backbone.** EfficientNet-B3 scored below B0; at this dataset size
  it overfits.
* **Longer training.** Both backbones peak by epoch 3 to 6 and then decay. The
  shipped recipe is 8 epochs.
* **Tuning the decision threshold.** The validation-selected threshold moved the
  test score by less than the noise of a 351-image test set, so the default 0.5
  is used.

### A second attempt at the recipe, and where the ceiling actually is

A later pass kept the algorithm identical (fine-tuned EfficientNet-B0) and
changed only the methodology: 299px instead of 224, mixup and cutmix instead of
early stopping, EMA weight averaging, and warmup before the cosine schedule.
That fixed the training pathology; runs improved monotonically instead of
peaking at epoch 3 and decaying, and per-model ranking quality rose clearly.

| Ensemble | Test accuracy | Test ROC AUC |
|---|---|---|
| v1 recipe, 5 seeds @224px (shipped) | 89.46% | 0.9608 |
| v2 recipe, 5 seeds @299px | 89.46% | 0.9653 |
| Both families combined, 10 models, multi-scale | 88.89% | 0.9669 |

Single-model AUC went from 0.933-0.951 (v1) to 0.958-0.963 (v2). So the better
recipe is genuinely a better *ranker*, and yet accuracy did not move: all three
ensembles land within about two images of each other on a 351-image test set,
which is inside the noise.

That is explained by the ROC ceiling. Sweeping every possible decision
threshold and keeping the best score *on the test labels themselves* (an oracle,
not a claimable number) gives at most **90.60%** for the strongest ensemble.
The shipped model already scores 89.46%, so under a point of headroom remains,
and none of it is reachable without cheating. Beating that needs a better model,
not better calibration, and higher resolution, heavier regularisation, weight
averaging, more members and multi-scale ensembling have all now been spent
on it.

### Why not 95%

95% on this task is above what published work reaches for a lesion-disjoint,
class-balanced benign vs malignant split, and none of the routes above get
close. Three independent signals say the remaining gap is the task, not the
engineering: the handcrafted learning curve is flat (five times the data bought
3.4 points), extra data actively hurts, and a fine-tuned CNN lands at 89%.

The ways a 95% number is usually produced on HAM10000 are all measurement
artifacts, and none of them reach 95% here either:

* **Lesion leakage.** Scoring on a random image split rather than a
  lesion-disjoint one is worth about 1.1 points (79.72% vs 78.63%).
* **Natural prevalence.** HAM10000 is 84.35% benign, so "always benign" already
  scores 84.35%. Optimising plain accuracy at natural prevalence reached 88.31%
  but collapsed recall on malignant to 49.1%, which is half the cancers missed.

---

## System Architecture & Workspace

* `app.py`: Flask app; loads `model_*.onnx` and averages them.
* `features.py`: the shared pre-processing pipeline and the exact model input
  tensor. Imported by both training and serving. Run it directly for a self-check.
* `train_classifier.py`: fine-tunes EfficientNet, writes `model.pt` and `model.onnx`.
* `download_data.py`: pulls HAM10000 from Kaggle into `dataset/`.
* `model_1.onnx` ... `model_5.onnx`: the deployed ensemble.
* `requirements.txt`: serving dependencies. `requirements-train.txt`: adds torch.

---

## Reproducing the model

```bash
pip install -r requirements-train.txt
python download_data.py                      # HAM10000 -> dataset/<class>/
python train_classifier.py                   # writes model.pt + model.onnx
```

`download_data.py` needs a Kaggle API token. Put it at `~/.kaggle/kaggle.json`
with `chmod 600`, **not** in the project root: a copy here gets swept into any
zip or commit of this directory, which is how API keys leak. A root-level
`kaggle.json` is still honoured if one exists, and is gitignored.

Useful environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `BACKBONE` | `b0` | `b0` (224px) or `b3` (300px) |
| `EPOCHS` | `8` | what the shipped models used; both backbones peak early |
| `EXTRA` | unset | CSV of extra `path,label` training rows |

To rebuild the shipped ensemble, run the trainer five times with `BACKBONE=b0
EPOCHS=8`, saving each `model.onnx` as `model_1.onnx` through `model_5.onnx`.
`app.py` picks up every `model_[0-9]*.onnx` it finds and averages them; with a
single `model.onnx` it serves that instead.

---

## Deployment notes

Measured on a fresh install of `requirements.txt` alone, serving the committed
`model_1..5.onnx` under `gunicorn app:app` with one worker:

| | Value |
|---|---|
| Resident memory, at boot | 202 MB |
| Resident memory, after 26 requests | 287 MB (flat) |
| Latency | ~350 ms/request locally, slower on a shared CPU |
| Deployed size | 79 MB, almost all of it the five models |

That fits a 512 MB instance, but only because `app.py` builds each ONNX session
with `enable_cpu_mem_arena = False` and a single intra-op thread. With
onnxruntime's defaults the same five models start at 482 MB and reach 726 MB by
the third request, because every session keeps a private arena that grows and
is never returned. If you re-enable the arena, or add ensemble members, expect
the worker to be OOM-killed. `ENSEMBLE_SIZE=3` trims memory and latency further
at a small accuracy cost.

Python must be 3.11 or newer (`render.yaml` pins 3.11.9): onnxruntime publishes
no 3.10 wheels past 1.16.3, and the build uses `--only-binary=:all:`.

---

## Local Installation & Setup

```bash
git clone https://github.com/AayushiKumari-tech/DermAlert-Framework.git
cd DermAlert-Framework
pip install -r requirements.txt
python app.py
```

## Author

* Developed by **Aayushi Kumari** - B.Tech CSE (6th Sem)
