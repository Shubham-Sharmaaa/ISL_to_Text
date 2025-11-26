import argparse
import os
import time
import cv2
import mediapipe as mp
import numpy as np
from joblib import load
from utils.hand_features import landmarks_to_feature
import json

def load_model_and_meta(model_path):
    clf = load(model_path)

    # Load metadata (class ordering)
    meta_path = model_path.replace(".joblib", ".meta.json")
    le = None
    labels = None

    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            labels = meta.get("classes", None)
        except Exception:
            labels = None

    # Load label encoder if present
    encoder_path = model_path.replace(".joblib", "_label_encoder.joblib")
    if os.path.exists(encoder_path):
        try:
            le = load(encoder_path)
            labels = le.classes_.tolist()
        except Exception:
            le = None

    # Fallback to model's internal classes_
    if labels is None and hasattr(clf, "classes_"):
        labels = clf.classes_.tolist()

    return clf, le, labels


def main():
    parser = argparse.ArgumentParser(description="Real-time ISL → Text inference (RandomForest).")
    parser.add_argument("--model", default="models/isl_rf.joblib", help="Path to trained model")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--conf-threshold", type=float, default=0.5, help="Minimum probability for prediction")
    parser.add_argument("--allow", nargs="*", default=None, help="Restrict to selected labels (optional)")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print("Model not found:", args.model)
        return

    clf, le, labels = load_model_and_meta(args.model)
    print("Loaded model:", args.model)
    print("Detected labels:", labels)
    if le:
        print("Label encoder loaded.")

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    sentence = ""
    last_pred = None
    stable_count = 0
    STABLE_N = 7   # number of frames needed to confirm a gesture

    frame_count = 0
    t_start = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            h, w = frame.shape[:2]

            pred_label = None
            pred_conf = 0.0

            # ------------ HAND DETECTION ------------
            if res.multi_hand_landmarks:
                for hand_landmarks in res.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # FIX: landmarks_to_feature returns (feats, scale)
                    result = landmarks_to_feature(hand_landmarks.landmark)
                    if isinstance(result, tuple):
                        feats_array, scale = result
                    else:
                        feats_array = result
                        scale = None

                    feats = np.asarray(feats_array).reshape(1, -1).astype(np.float32)

                    # Skip frames where the hand is too small / far
                    if scale is not None and scale < 0.02:
                        pred_label = None
                        pred_conf = 0.0
                        break

                    # ------------ PREDICTION ------------
                    if hasattr(clf, "predict_proba"):
                        probs = clf.predict_proba(feats)[0]
                        idx = int(np.argmax(probs))
                        pred_label = labels[idx]
                        pred_conf = float(probs[idx])
                    else:
                        pred_label = clf.predict(feats)[0]
                        pred_conf = 1.0

                    # Restrict labels (optional)
                    if args.allow is not None and pred_label not in args.allow:
                        pred_label = None
                        pred_conf = 0.0

                    # ------------ DEBOUNCE LOGIC ------------
                    if pred_label and pred_conf >= args.conf_threshold:
                        if pred_label == last_pred:
                            stable_count += 1
                        else:
                            stable_count = 1
                            last_pred = pred_label

                        if stable_count >= STABLE_N:
                            if pred_label == "SPACE":
                                sentence += " "
                            elif pred_label == "DEL":
                                sentence = sentence[:-1]
                            else:
                                sentence += pred_label

                            stable_count = 0
                    else:
                        stable_count = 0
                        last_pred = None

                    break  # only first detected hand processed

            # ------------ FPS ------------
            frame_count += 1
            if frame_count % 30 == 0:
                t_now = time.time()
                fps = 30.0 / (t_now - t_start) if t_now > t_start else 0.0
                t_start = t_now
            else:
                fps = None

            # ------------ UI OVERLAY ------------
            text = f"Pred: {pred_label or '-'} ({pred_conf:.2f})"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

            if fps is not None:
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)

            cv2.rectangle(frame, (10, h-60), (w-10, h-10), (50,50,50), 1)
            cv2.putText(frame, sentence[-60:], (20, h-25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

            cv2.imshow("ISL → Text (press q to quit, c to clear)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('c'):
                sentence = ""

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
