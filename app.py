import os
import glob
import time

import cv2
import numpy as np
import onnxruntime as ort
from flask import Flask, render_template, request, redirect

import features

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ONNX rather than the trained .pt on purpose: onnxruntime is a ~15 MB wheel
# against torch's several hundred, which is the difference between fitting a
# small instance and not.
#
# An ensemble of independently trained members, averaged. Five seeds of the
# same recipe score 89.5% on the held-out test set against 84.9-88.0% for the
# individual members, because their errors are only partly correlated.
# Falls back to a single model.onnx, which is what train_classifier.py writes.
MODEL_PATHS = sorted(glob.glob("model_[0-9]*.onnx")) or (
    ["model.onnx"] if os.path.exists("model.onnx") else [])
MODEL_PATHS = MODEL_PATHS[:int(os.environ.get("ENSEMBLE_SIZE", 5))]


def _session(path):
    """One session, tuned for a small container rather than for throughput.

    Each session otherwise keeps its own CPU arena, and those arenas grow with
    every request and are never handed back. Five of them put the worker at
    482 MB on the first request and 726 MB by the third, which OOMs a 512 MB
    instance. Disabling the arena trades a little speed for a flat, predictable
    footprint; one intra-op thread avoids a per-thread arena on top of that.
    """
    opts = ort.SessionOptions()
    opts.enable_cpu_mem_arena = False
    opts.enable_mem_pattern = False
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(path, sess_options=opts,
                                providers=["CPUExecutionProvider"])


sessions = [_session(p) for p in MODEL_PATHS]
if sessions:
    # Read the input size off the graph instead of hardcoding it, so swapping
    # backbones in train_classifier.py cannot silently desync the server.
    MODEL_SIZE = sessions[0].get_inputs()[0].shape[-1]
    print(f"--> {len(sessions)} model(s) loaded ({', '.join(MODEL_PATHS)}), "
          f"expecting {MODEL_SIZE}x{MODEL_SIZE} input.")
else:
    MODEL_SIZE = 224
    print("--> Warning: no model found. Run 'python train_classifier.py'.")


def predict(batch):
    """Class probabilities, averaged over 4 flips and every ensemble member.

    Test-time augmentation is not decoration here: the training loop selected
    its epoch on 4-view TTA scores, so serving a single view would report a
    different model than the one that was measured. Softmax is taken per model
    before averaging, matching how the ensemble was evaluated.
    """
    views = np.ascontiguousarray(
        np.concatenate([batch, batch[:, :, :, ::-1], batch[:, :, ::-1, :],
                        batch[:, :, ::-1, ::-1]]))
    probs = []
    for sess in sessions:
        logits = sess.run(None, {sess.get_inputs()[0].name: views})[0].mean(axis=0)
        exp = np.exp(logits - logits.max())
        probs.append(exp / exp.sum())
    return np.mean(probs, axis=0)


@app.route('/', methods=['GET', 'POST'])
def upload_page():
    if request.method != 'POST':
        return render_template('index.html', deployed=False)

    if 'file' not in request.files or request.files['file'].filename == '':
        return redirect(request.url)
    file = request.files['file']

    for stale in glob.glob(os.path.join(app.config['UPLOAD_FOLDER'], '*')):
        try:
            os.remove(stale)
        except OSError:
            pass

    run_id = str(int(time.time()))
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], f'input_{run_id}.jpg')
    file.save(input_path)

    prepared = features.prepare(input_path, MODEL_SIZE) if sessions else None
    if prepared is None:
        return render_template(
            'index.html', deployed=True,
            condition="Execution Error", confidence="0.0%",
            triage="Could not read that file as an image, or model.onnx is missing.",
            raw_img="", enhanced_img="", shaved_img="")

    (resized, enhanced, shaved), batch = prepared
    names = [f"{stage}_{run_id}.jpg" for stage in ("raw", "enhanced", "shaved")]
    for name, image in zip(names, (resized, enhanced, shaved)):
        cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], name), image)

    probs = predict(batch)
    idx = int(probs.argmax())
    confidence = f"{probs[idx] * 100:.1f}%"

    if idx == 0:
        condition = "Benign Skin Structure"
        triage = ("The classifier places this lesion on the benign side of its "
                  "decision boundary. On a held-out test set it is right about "
                  "89% of the time and misses roughly 1 malignant lesion in 8, "
                  "so this is a screening aid and not a diagnosis. Anything "
                  "that changes in size, shape or colour should be looked at by "
                  "a dermatologist regardless of this result.")
    else:
        condition = "Inflamed/Malignant Lesion Detected"
        triage = ("The classifier places this lesion on the malignant side of "
                  "its decision boundary. It was trained on dermoscopic images, "
                  "so ordinary phone photos, acne, and strong lighting can push "
                  "it this way. This is not a diagnosis: please have a "
                  "dermatologist review the lesion.")

    return render_template('index.html', deployed=True,
                           condition=condition, confidence=confidence, triage=triage,
                           raw_img=names[0], enhanced_img=names[1], shaved_img=names[2])


if __name__ == '__main__':
    app.run(debug=True, port=5000)
