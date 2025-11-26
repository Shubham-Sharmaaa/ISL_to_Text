import numpy as np

def landmarks_to_feature(landmarks):
    """
    Convert 21 hand landmarks (x,y,z normalized coords) to a translation/scale-invariant feature vector.
    landmarks: list of 21 landmark objects or dicts with .x, .y, .z
    Returns:
      feats: (63,) numpy array (flattened: x0,y0,z0,x1,y1,z1,...)
      scale: float (max distance used for normalization) -- useful to reject tiny/too-far hands
    Notes:
      - Assumes index 0 is the wrist (Mediapipe standard).
      - If landmarks is not length 21 or contains NaNs, raises ValueError.
    """
    if landmarks is None:
        raise ValueError("landmarks is None")
    if len(landmarks) != 21:
        raise ValueError(f"Expected 21 landmarks, got {len(landmarks)}")

    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)

    if np.isnan(pts).any():
        raise ValueError("NaN in landmark coordinates")

    # Translate so that wrist (index 0) is origin
    origin = pts[0:1, :].copy()
    pts = pts - origin

    # Scale by max distance to avoid division by zero
    dists = np.linalg.norm(pts, axis=1)
    scale = float(dists.max())
    if scale < 1e-6:
        scale = 1.0
    pts = pts / scale

    # Flatten to 63-d vector
    feats = pts.flatten()
    return feats, scale


# Backwards-compatible wrapper if code expects only features:
def landmarks_to_feature_flat(landmarks):
    feats, _ = landmarks_to_feature(landmarks)
    return feats

# Default labels list (only used for convenience in collection UI)
LABELS_DEFAULT = [
    "A","B","C","D","E","F","G","H","I","L","O","S","SPACE","DEL"
]
