import os
import json
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from joblib import load
from utils.hand_features import landmarks_to_feature
from flask_cors import CORS       # NEW

app = Flask(
    __name__,
    static_folder="frontend",
    static_url_path=""
)
CORS(app)  
MODEL_PATH = "models/isl_rf.joblib"
ENC_PATH = MODEL_PATH.replace(".joblib", "_label_encoder.joblib")

clf = None
labels = None

# ------------ Load model + labels ------------
if os.path.exists(MODEL_PATH):
    clf = load(MODEL_PATH)
    print("[app] Loaded model:", MODEL_PATH)
else:
    print("[app] WARNING: model not found:", MODEL_PATH)

if clf is not None and os.path.exists(ENC_PATH):
    try:
        le = load(ENC_PATH)
        labels = le.classes_.tolist()
        print("[app] Loaded label encoder with classes:", labels)
    except Exception as e:
        print("[app] Could not load label encoder:", e)
        labels = getattr(clf, "classes_", None)
        if labels is not None:
            labels = labels.tolist()
else:
    if clf is not None and hasattr(clf, "classes_"):
        labels = clf.classes_.tolist()


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


def json_landmarks_to_feature(lm_json):
    """
    Convert JS landmarks (21 x [x,y,z]) into the same feature vector
    used in collect_data.py + infer_realtime.py.
    """
    if not isinstance(lm_json, (list, tuple)) or len(lm_json) != 21:
        raise ValueError("Expected 21 landmarks, got %r" % len(lm_json))

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
            x = float(item[0])
            y = float(item[1])
            z = float(item[2]) if len(item) > 2 else 0.0
        else:
            raise ValueError("Invalid landmark format")

  
        x = 1.0 - x

        pts.append(LM(x, y, z))

    res = landmarks_to_feature(pts)
    if isinstance(res, tuple):
        feats, scale = res
    else:
        feats, scale = res, None

    feats = np.asarray(feats, dtype=np.float32).reshape(1, -1)
    return feats


@app.route("/predict", methods=["POST"])
def predict():
    if clf is None:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.get_json(silent=True)
    if not data or "landmarks" not in data:
        return jsonify({"error": "No landmarks provided"}), 400

    try:
        feats = json_landmarks_to_feature(data["landmarks"])
    except Exception as e:
        print("[app] Feature error:", e)
        return jsonify({"error": "Feature error: " + str(e)}), 400

    try:
        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(feats)[0]
            idx = int(np.argmax(probs))
            label = labels[idx] if labels is not None else str(idx)
            conf = float(probs[idx])
        else:
            label = clf.predict(feats)[0]
            conf = 1.0
    except Exception as e:
        print("[app] Prediction failed:", e)
        return jsonify({"error": "Prediction failed: " + str(e)}), 500

    print(f"[app] Pred: {label} ({conf:.2f})")
    return jsonify({"label": label, "confidence": conf})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
