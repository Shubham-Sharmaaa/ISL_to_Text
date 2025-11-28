import numpy as np

def landmarks_to_feature(landmarks):
    """
    Convert 21 hand landmarks (x,y,z normalized coords) to a translation/scale-invariant feature vector.
    landmarks: list of 21 landmark objects or dicts with .x, .y, .z
    Returns:
      feats: (63,) numpy array (flattened: x0,y0,z0,x1,y1,z1,...)
      scale: float (max distance used for normalization) -- useful to reject tiny/too-far hands
    """
    if landmarks is None:
        raise ValueError("landmarks is None")
    if len(landmarks) != 21:
        raise ValueError(f"Expected 21 landmarks, got {len(landmarks)}")

    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)

    if np.isnan(pts).any():
        raise ValueError("NaN in landmark coordinates")

   
    origin = pts[0:1, :].copy()
    pts = pts - origin

    dists = np.linalg.norm(pts, axis=1)
    scale = float(dists.max())
    if scale < 1e-6:
        scale = 1.0
    pts = pts / scale

 
    feats = pts.flatten()
    return feats, scale


def landmarks_to_feature_flat(landmarks):
    feats, _ = landmarks_to_feature(landmarks)
    return feats


LABELS_DEFAULT = [
    "A","B","C","D","E","F","G","H","I","J","K","L","M",
    "N","O","P","Q","R","S","T","U","V","W","X","Y","Z",
    "SPACE","DEL"
]

