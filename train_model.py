# import os
# import argparse
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.svm import SVC
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import classification_report, accuracy_score
# from joblib import dump

# def main():
#     parser = argparse.ArgumentParser(description="Train ISL gesture classifier from CSV.")
#     parser.add_argument("--csv", default="data/isl_data.csv", help="Path to collected CSV")
#     parser.add_argument("--model", default="models/isl_svc.joblib", help="Output model path")
#     args = parser.parse_args()

#     if not os.path.exists(args.csv):
#         print(f"CSV not found: {args.csv}")
#         return

#     import pandas as pd
#     df = pd.read_csv(args.csv)
#     X = df.drop(columns=["label"]).values
#     y = df["label"].values

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)

#     pipe = Pipeline([
#         ("scaler", StandardScaler()),
#         ("clf", SVC(kernel="rbf", C=10, gamma="scale", probability=True, class_weight="balanced"))
#     ])

#     pipe.fit(X_train, y_train)
#     y_pred = pipe.predict(X_test)
#     acc = accuracy_score(y_test, y_pred)
#     print("Accuracy:", acc)
#     print(classification_report(y_test, y_pred))

#     os.makedirs(os.path.dirname(args.model), exist_ok=True)
#     dump(pipe, args.model)
#     print("Saved model to", args.model)

# if __name__ == "__main__":
#     main()
import os
import argparse
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from joblib import dump

def main():
    parser = argparse.ArgumentParser(description="Train ISL gesture classifier using Random Forest.")
    parser.add_argument("--csv", default="data/isl_data.csv", help="Path to CSV dataset")
    parser.add_argument("--model", default="models/isl_rf.joblib", help="Output model path")
    parser.add_argument("--n-estimators", type=int, default=200, help="Number of trees")
    parser.add_argument("--test-size", type=float, default=0.15, help="Test split fraction")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"Error: CSV not found → {args.csv}")
        return

    df = pd.read_csv(args.csv)

    if "label" not in df.columns:
        raise SystemExit("ERROR: CSV does not contain a 'label' column.")

    # Features + labels
    X = df.drop(columns=["label"]).values
    y = df["label"].values

    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc,
        test_size=args.test_size,
        stratify=y_enc,
        random_state=args.random_state
    )

    # Pipeline
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=args.n_estimators,
            n_jobs=-1,
            random_state=args.random_state,
            class_weight="balanced_subsample"
        ))
    ])

    print("Training Random Forest...")
    pipe.fit(X_train, y_train)

    # Evaluation
    y_pred = pipe.predict(X_test)
    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save model
    os.makedirs(os.path.dirname(args.model), exist_ok=True)
    dump(pipe, args.model)

    # Save label encoder
    enc_path = args.model.replace(".joblib", "_label_encoder.joblib")
    dump(le, enc_path)

    # Save metadata
    meta = {
        "model_type": "RandomForest",
        "n_estimators": args.n_estimators,
        "classes": le.classes_.tolist(),
        "n_features": X.shape[1]
    }
    meta_path = args.model.replace(".joblib", ".meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved Model: {args.model}")
    print(f"Saved Encoder: {enc_path}")
    print(f"Saved Metadata: {meta_path}")

if __name__ == "__main__":
    main()
