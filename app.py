# app.py
from flask import Flask, render_template, request, jsonify
from joblib import load
import numpy as np
import os
import json
from utils.hand_features import landmarks_to_feature  # your module

app = Flask(__name__, static_folder="static", template_folder="templates")

MODEL_PATH = "models/isl_rf.joblib"
ENC_PATH = "models/isl_rf_label_encoder.joblib"

# load model and encoder once
clf = None
labels = None
if os.path.exists(MODEL_PATH):
    clf = load(MODEL_PATH)
if os.path.exists(ENC_PATH):
    try:
        le = load(ENC_PATH)
        labels = le.classes_.tolist()
    except Exception:
        labels = getattr(clf, "classes_", None).tolist() if clf is not None and hasattr(clf, "classes_") else None
else:
    labels = getattr(clf, "classes_", None).tolist() if clf is not None and hasattr(clf, "classes_") else None


@app.route("/")
def index():
    return render_template("index.html")


def json_landmarks_to_feature(lm_json):
    """
    lm_json expected: list of 21 items, each is [x, y, z] OR {"x":..., "y":..., "z":...}
    Returns: 1D numpy array of shape (63,) (feature vector) or raises ValueError
    """
    if not isinstance(lm_json, (list, tuple)) or len(lm_json) != 21:
        raise ValueError("Expected 21 landmarks")

    # Convert to objects compatible with landmarks_to_feature (which expects objects with .x,.y,.z OR we adapt)
    # We'll create a simple object with attributes:
    class LM:
        def __init__(self, x, y, z):
            self.x = x
            self.y = y
            self.z = z

    pts = []
    for item in lm_json:
        if isinstance(item, dict):
            x = float(item.get("x", 0.0))
            y = float(item.get("y", 0.0))
            z = float(item.get("z", 0.0))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            x = float(item[0]); y = float(item[1]); z = float(item[2]) if len(item) > 2 else 0.0
        else:
            raise ValueError("landmark format invalid")
        pts.append(LM(x, y, z))

    res = landmarks_to_feature(pts)
    # landmarks_to_feature may return (feats, scale) or feats
    if isinstance(res, tuple):
        feats, scale = res
    else:
        feats = res
    feats = np.asarray(feats, dtype=np.float32).reshape(-1)
    return feats


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accept JSON:
    {
      "landmarks": [[x,y,z], ... 21 items],
      "request_type": "auto" (optional)
    }
    Response:
    { "label": "A", "confidence": 0.98 }
    """
    global clf, labels
    data = request.get_json(force=True)
    if data is None or "landmarks" not in data:
        return jsonify({"error": "No landmarks provided"}), 400
    try:
        feats = json_landmarks_to_feature(data["landmarks"]).reshape(1, -1)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    if clf is None:
        return jsonify({"error": "Model not loaded on server"}), 500

    try:
        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(feats)[0]
            idx = int(np.argmax(probs))
            label = labels[idx] if labels is not None else str(idx)
            conf = float(probs[idx])
        else:
            pred = clf.predict(feats)[0]
            label = str(pred)
            conf = 1.0
    except Exception as e:
        return jsonify({"error": "Model prediction failed: " + str(e)}), 500

    return jsonify({"label": label, "confidence": conf})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
