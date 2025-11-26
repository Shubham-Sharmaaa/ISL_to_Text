# import os
# import csv
# import time
# import argparse
# import cv2
# import mediapipe as mp
# import numpy as np
# from utils.hand_features import landmarks_to_feature, LABELS_DEFAULT

# def main():
#     parser = argparse.ArgumentParser(description="Collect ISL hand landmark data per label.")
#     parser.add_argument("--out", default="data/isl_data.csv", help="CSV output path")
#     parser.add_argument("--labels", nargs="*", default=LABELS_DEFAULT, help="Labels to cycle through")
#     parser.add_argument("--samples-per-label", type=int, default=200, help="Number of samples per label")
#     parser.add_argument("--camera", type=int, default=0, help="Camera index")
#     args = parser.parse_args()

#     os.makedirs(os.path.dirname(args.out), exist_ok=True)

#     mp_hands = mp.solutions.hands
#     hands = mp_hands.Hands(
#         static_image_mode=False,
#         max_num_hands=1,
#         model_complexity=1,
#         min_detection_confidence=0.6,
#         min_tracking_confidence=0.6,
#     )
#     mp_drawing = mp.solutions.drawing_utils

#     cap = cv2.VideoCapture(args.camera)
#     if not cap.isOpened():
#         print("Error: Could not open camera")
#         return

#     # Prepare CSV
#     header = ["label"] + [f"f{i}" for i in range(63)]
#     new_file = not os.path.exists(args.out)
#     f = open(args.out, "a", newline="")
#     writer = csv.writer(f)
#     if new_file:
#         writer.writerow(header)

#     try:
#         for label in args.labels:
#             count = 0
#             print(f"Get ready to record label: {label}")
#             time.sleep(2)
#             while count < args.samples_per_label:
#                 ret, frame = cap.read()
#                 if not ret:
#                     break
#                 frame = cv2.flip(frame, 1)
#                 rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 res = hands.process(rgb)

#                 h, w = frame.shape[:2]
#                 info = f"Label: {label}  Sample: {count}/{args.samples_per_label}"
#                 cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

#                 if res.multi_hand_landmarks:
#                     for hand_landmarks in res.multi_hand_landmarks:
#                         # draw landmarks for feedback
#                         mp_drawing.draw_landmarks(
#                             frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
#                         )
#                         feats = landmarks_to_feature(hand_landmarks.landmark)
#                         writer.writerow([label] + feats.tolist())
#                         count += 1
#                         break  # only one hand
#                 cv2.imshow("Collect ISL Data (press q to quit)", frame)
#                 if cv2.waitKey(1) & 0xFF == ord('q'):
#                     print("Quitting...")
#                     return
#             print(f"Done label: {label}")
#         print("Data collection completed.")
#     finally:
#         f.close()
#         cap.release()
#         cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()
import os
import csv
import time
import argparse
import cv2
import mediapipe as mp
import numpy as np
from utils.hand_features import landmarks_to_feature, LABELS_DEFAULT

def make_header():
    # produce header: x0,y0,z0,x1,y1,z1,...,x20,y20,z20,label
    header = []
    for i in range(21):
        header += [f"x{i}", f"y{i}", f"z{i}"]
    header.append("label")
    return header

def main():
    parser = argparse.ArgumentParser(description="Collect ISL hand landmark data per label.")
    parser.add_argument("--out", default="data/isl_data.csv", help="CSV output path")
    parser.add_argument("--labels", nargs="*", default=LABELS_DEFAULT, help="Labels to cycle through")
    parser.add_argument("--samples-per-label", type=int, default=200, help="Number of samples per label")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--min-scale", type=float, default=0.06, help="Minimum normalized hand scale to accept sample")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

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
        print("Error: Could not open camera")
        return

    header = make_header()
    new_file = not os.path.exists(args.out)
    f = open(args.out, "a", newline="")
    writer = csv.writer(f)
    if new_file:
        writer.writerow(header)

    try:
        for label in args.labels:
            count = 0
            print(f"Get ready to record label: {label}")
            # give user 2 seconds to prepare; during this time show text on screen too
            countdown = 2
            for t in range(countdown, 0, -1):
                print(f"Starting in {t}...")
                time.sleep(1)

            while count < args.samples_per_label:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = hands.process(rgb)

                h, w = frame.shape[:2]
                info = f"Label: {label}  Sample: {count}/{args.samples_per_label}  (press 's' to skip label, 'q' to quit)"
                cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

                if res.multi_hand_landmarks:
                    for hand_landmarks in res.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        try:
                            feats, scale = landmarks_to_feature(hand_landmarks.landmark)
                        except Exception as e:
                            # skip invalid frames
                            print("Invalid landmarks:", e)
                            continue

                        # Reject very small / far-away hands
                        if scale < args.min_scale:
                            cv2.putText(frame, "Hand too small — move closer", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                        else:
                            # write row with label at the end (header: x0,y0,...,z20,label)
                            row = feats.tolist() + [label]
                            writer.writerow(row)
                            f.flush()
                            count += 1
                        break  # only one hand processed

                cv2.imshow("Collect ISL Data (press q to quit, s to skip label)", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Quitting...")
                    return
                if key == ord('s'):
                    print("Skipping current label...")
                    break  # move to next label

            print(f"Done label: {label}  Collected: {count}")
        print("Data collection completed.")
    finally:
        f.close()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
