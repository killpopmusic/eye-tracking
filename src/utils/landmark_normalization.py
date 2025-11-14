import numpy as np

def normalize_landmarks(
  landmarks: np.ndarray,
  key_indices=None,
  mode: str = "raw",
  stable_indices=None,
  eye_corner_indices=(33, 263),
  eye_dist_outlier_factor: float = 3.0,
):
  """Normalize facial landmarks with configurable steps 

  Modes:
    raw            -> no normalization
    center         -> subtract head center
    center_scale   -> center + scale by eye distance
    center_rotate  -> center + rotate (no scale)
    full (default) -> center + rotate + scale
  """

  if stable_indices is None:
    stable_indices = [1, 6, 152, 10, 9]

  landmarks_xy = landmarks[:, :, :2]

  if mode == "raw":
    out = landmarks_xy
  else:
    # Centering
    head_center = landmarks_xy[:, stable_indices, :].mean(axis=1, keepdims=True)
    centered = landmarks_xy - head_center

    r_idx, l_idx = eye_corner_indices
    right_outer = landmarks_xy[:, r_idx, :]
    left_outer = landmarks_xy[:, l_idx, :]
    eye_vec = left_outer - right_outer
    eye_dist = np.linalg.norm(eye_vec, axis=1, keepdims=True)
    eye_dist[eye_dist < 1e-8] = 1e-8

    # Orientation from instantaneous eye vector (no temporal smoothing)
    cos_t_all = eye_vec[:, 0:1] / eye_dist
    sin_t_all = eye_vec[:, 1:2] / eye_dist

    median_dist = np.median(eye_dist)
    too_large = eye_dist > median_dist * eye_dist_outlier_factor
    too_small = eye_dist < median_dist / eye_dist_outlier_factor
    if np.any(too_large) or np.any(too_small):
      eye_dist[too_large | too_small] = median_dist

    R = np.stack([
      np.concatenate([cos_t_all, -sin_t_all], axis=1),
      np.concatenate([sin_t_all,  cos_t_all], axis=1)
    ], axis=1)
    rotated = np.einsum('nij,nkj->nki', R, centered)

    if mode == "center":
      out = centered
    elif mode == "center_rotate":
      out = rotated
    elif mode == "center_scale":
      scale = eye_dist[:, :, None]
      scale[scale < 1e-8] = 1e-8
      out = centered / scale
    else:  # full
      scale = eye_dist[:, :, None]
      scale[scale < 1e-8] = 1e-8
      out = rotated / scale

  if key_indices is not None:
    return out[:, key_indices, :]
  return out
