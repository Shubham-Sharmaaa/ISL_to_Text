# Indian Sign Language (ISL) → Text (Working Prototype)

A minimal, **working end-to-end project** that lets you:

1. **Collect** your own ISL samples using MediaPipe hand landmarks
2. **Train** a classifier (SVM) on those samples
3. **Run** real-time inference from your webcam to output text

This aligns with your Software Engineering report and can be extended into a production app.

---

## Quick Start

### 1) Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt
```

> Ensure your environment has a working webcam and a recent Python (3.9+).

### 2) Collect Data

Decide which labels (gestures) you want. A default label set is:

```
A B C D E F G H I L O S SPACE DEL
```

Run collection (press `q` to quit anytime):

```bash
python collect_data.py --samples-per-label 200
# or customize:
python collect_data.py --labels A B C SPACE DEL --samples-per-label 250
```

This creates/append to `data/isl_data.csv`. For each label, hold the gesture steady while the window shows it’s recording.

**Tips**

- Use consistent lighting and camera position.
- Capture from both slight rotations to add robustness.
- If you see mislabels, you can delete rows from the CSV and recollect.

### 3) Train

```bash
python train_model.py --csv data/isl_data.csv --model models/isl_svc.joblib
```

You’ll see accuracy and a classification report. Improve it by:

- Adding more samples
- Ensuring stable poses
- Increasing `--samples-per-label`

### 4) Run Real-time Inference

```bash
python infer_realtime.py --model models/isl_svc.joblib
```

- Gesture is **debounced** using a stability counter to avoid noisy appends.
- Special labels:
  - `SPACE` → adds a space
  - `DEL` → deletes last character

Options:

```bash
python infer_realtime.py --conf-threshold 0.6 --allow A B C SPACE DEL
```

---

## How It Works

- **MediaPipe Hands** provides 21 hand landmarks → we normalize by wrist origin and max distance to make them position/scale invariant.
- The 63-dim feature vector feeds a **scikit-learn SVM**.
- No heavy GPU training required; it’s fast on CPU.

**Folders**

- `data/` → `isl_data.csv` grows as you collect
- `models/` → saved `.joblib` model
- `utils/hand_features.py` → feature engineering & default label list

---

## Extending to Full ISL Alphabet & Words

- Add more labels into `LABELS_DEFAULT` in `utils/hand_features.py`
- Collect more data per label (≥500) from multiple users
- Consider a **temporal model** (LSTM/1D-CNN/Transformer) for dynamic gestures.
- Add a **language model** to autocorrect sequences into words.
- Build a **web UI** with FastAPI + React for deployment.

---

## Troubleshooting

- **No camera**: change `--camera` index (0,1,2...).
- **Low accuracy**: collect more/clean data, improve lighting, keep the hand centered.
- **Wrong predictions repeating**: increase stability window in `infer_realtime.py` (STABLE_N).

---

## Credits

- Uses [MediaPipe Hands] for landmark detection.
- You own the dataset you collect; keep it private if needed.

---

## License

MIT (for the provided code). Your collected data remains yours.
