"""Fetch HAM10000 from Kaggle and lay it out as dataset/<class>/ for train_classifier.py.

Two stages, both idempotent:
  1. download  -> dataset_raw/  (HAM10000_images_part_{1,2}/ + HAM10000_metadata.csv)
  2. organise  -> dataset/benign/ and dataset/malignant_inflamed/ (symlinks, no extra disk)

Only one image per lesion_id is kept. HAM10000 ships 10015 images of just 7470
distinct lesions; keeping the duplicates lets the same lesion land in both the
train and the test split, which inflates the reported accuracy without making
the model any better.
"""
import os
import csv
from collections import defaultdict

# Only point Kaggle at this folder when a token actually lives here. The token
# belongs in ~/.kaggle/kaggle.json: a copy in the project root gets swept into
# any zip or commit of this directory, which is how API keys leak.
if os.path.exists(os.path.join(os.getcwd(), "kaggle.json")):
    os.environ.setdefault("KAGGLE_CONFIG_DIR", os.getcwd())

RAW_DIR = "dataset_raw"
OUT_DIR = "dataset"
IMAGE_DIRS = ("HAM10000_images_part_1", "HAM10000_images_part_2")
METADATA = "HAM10000_metadata.csv"

# HAM10000 dx codes grouped into the two folders train_classifier.py expects.
# malignant/pre-malignant: melanoma, basal cell carcinoma, actinic keratoses.
CLASS_OF_DX = {
    "nv": "benign",
    "bkl": "benign",
    "df": "benign",
    "vasc": "benign",
    "mel": "malignant_inflamed",
    "bcc": "malignant_inflamed",
    "akiec": "malignant_inflamed",
}

print("=========================================")
print("  DermAlert Automatic Dataset Builder    ")
print("=========================================")


def download():
    if os.path.exists(os.path.join(RAW_DIR, METADATA)):
        print(f"-> {RAW_DIR}/ already populated, skipping download.")
        return
    try:
        import kaggle
    except Exception:
        raise SystemExit("Authentication Error: drop a valid 'kaggle.json' into this folder.")
    print("-> Downloading HAM10000 (~5 GB, several minutes)...")
    kaggle.api.dataset_download_files(
        "kmader/skin-cancer-mnist-ham10000", path=RAW_DIR, unzip=True
    )
    print("Success: HAM10000 pulled and unzipped.")


def locate_images():
    """image_id -> path on disk."""
    found = {}
    for sub in IMAGE_DIRS:
        folder = os.path.join(RAW_DIR, sub)
        if not os.path.isdir(folder):
            continue
        for name in os.listdir(folder):
            stem, ext = os.path.splitext(name)
            if ext.lower() in (".jpg", ".jpeg", ".png"):
                found[stem] = os.path.abspath(os.path.join(folder, name))
    return found


def organise():
    images = locate_images()
    if not images:
        raise SystemExit(f"No images under {RAW_DIR}/. Run the download stage first.")

    with open(os.path.join(RAW_DIR, METADATA), newline="") as fh:
        rows = list(csv.DictReader(fh))

    # One image per lesion: deterministic pick so reruns give the same dataset.
    by_lesion = defaultdict(list)
    for row in rows:
        by_lesion[row["lesion_id"]].append(row)

    counts = defaultdict(int)
    for label in set(CLASS_OF_DX.values()):
        folder = os.path.join(OUT_DIR, label)
        os.makedirs(folder, exist_ok=True)
        for stale in os.listdir(folder):
            os.remove(os.path.join(folder, stale))

    for lesion_id in sorted(by_lesion):
        row = sorted(by_lesion[lesion_id], key=lambda r: r["image_id"])[0]
        label = CLASS_OF_DX.get(row["dx"])
        src = images.get(row["image_id"])
        if label is None or src is None:
            continue
        os.symlink(src, os.path.join(OUT_DIR, label, f"{row['image_id']}.jpg"))
        counts[label] += 1

    for label, n in sorted(counts.items()):
        print(f"   {OUT_DIR}/{label}: {n} images")
    return counts


download()
print("\nOrganising into class folders (one image per lesion)...")
organise()
print("\n[Status] Dataset ready. Run: python train_classifier.py")
