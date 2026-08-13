"""Single source of truth for what the model sees.

train_classifier.py and app.py both import from here. They used to carry their
own copies of the pipeline and the copies had drifted -- training ran CLAHE +
bilateral(7) + MORPH_RECT(9,9) + fixed threshold while the live app ran a gamma
LUT + bilateral(5) + MORPH_RECT(7,7) + adaptive threshold -- so the deployed
model was fed vectors it had never trained on. Keeping one module is what stops
that recurring, and it matters more now: a CNN is even less forgiving of an
input distribution it did not train on.

What changed from the SVM era: the 26-slot handcrafted vector is gone. It was
the accuracy ceiling, not the tuning (see train_classifier.py's docstring).
CLAHE contrast enhancement and Dull-Razor hair removal are kept, unchanged --
they measurably neither help nor hurt the CNN, and keeping them means the three
processing stages the web UI displays are genuinely what the classifier sees.
"""
import cv2
import numpy as np

VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

# ImageNet statistics: the backbone is pretrained, so its normalisation is fixed.
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Images are prepared slightly larger than the network input so training can
# take random crops and inference a centre crop. Both sides must use the same
# ratio or the model sees a different field of view at serve time than it
# trained on.
CROP_RATIO = 1.14


def source_size(model_size):
    """Pixel size to prepare an image at, for a network expecting model_size."""
    return int(model_size * CROP_RATIO)


def preprocess(img, size):
    """Resize -> CLAHE contrast enhancement -> Dull-Razor hair removal.

    Returns (resized, enhanced, shaved) so the web UI can show each stage.
    """
    resized = cv2.resize(img, (size, size))

    ycrcb = cv2.cvtColor(resized, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    cl_y = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(5, 5)).apply(y)
    enhanced = cv2.cvtColor(cv2.merge((cl_y, cr, cb)), cv2.COLOR_YCrCb2BGR)

    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    smooth = cv2.bilateralFilter(gray, 7, 50, 50)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    blackhat = cv2.morphologyEx(smooth, cv2.MORPH_BLACKHAT, kernel)
    _, hair_mask = cv2.threshold(blackhat, 11, 255, cv2.THRESH_BINARY)
    shaved = cv2.inpaint(enhanced, hair_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    return resized, enhanced, shaved


def model_input(shaved, model_size):
    """Shaved BGR image -> the exact float32 NCHW batch the ONNX graph wants.

    Centre-crops to model_size, converts BGR->RGB, scales to 0..1 and applies
    the ImageNet normalisation.
    """
    h, w = shaved.shape[:2]
    top, left = (h - model_size) // 2, (w - model_size) // 2
    crop = shaved[top:top + model_size, left:left + model_size]

    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - np.float32(MEAN)) / np.float32(STD)
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])


def prepare(image_path, model_size):
    """Path -> (stages, batch), or None if the file is not a readable image."""
    if not image_path.lower().endswith(VALID_EXTENSIONS):
        return None
    img = cv2.imread(image_path)
    if img is None:
        return None
    try:
        stages = preprocess(img, source_size(model_size))
    except cv2.error:
        return None
    return stages, model_input(stages[2], model_size)


def _self_check():
    """The serving path and the training path must agree on every pixel."""
    import os
    import tempfile

    size = 300
    rng = np.random.RandomState(0)
    img = rng.randint(0, 255, (450, 600, 3), dtype=np.uint8)

    _, _, shaved = preprocess(img, source_size(size))
    assert shaved.shape == (source_size(size), source_size(size), 3), shaved.shape

    batch = model_input(shaved, size)
    assert batch.shape == (1, 3, size, size), batch.shape
    assert batch.dtype == np.float32, batch.dtype
    # Normalised ImageNet inputs land roughly in -2.2 .. 2.7; a raw 0..255
    # tensor leaking through would blow straight past this.
    assert -5 < batch.min() and batch.max() < 5, (batch.min(), batch.max())

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "probe.png")
        cv2.imwrite(path, img)
        via_path = prepare(path, size)
        assert via_path is not None
        assert np.allclose(batch, via_path[1]), "prepare/model_input disagree"
        assert prepare(os.path.join(tmp, "notes.txt"), size) is None

    print(f"features.py self-check OK: {batch.shape} input, both paths agree")


if __name__ == "__main__":
    _self_check()
