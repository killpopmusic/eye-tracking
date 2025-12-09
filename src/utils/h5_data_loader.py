import h5py
import numpy as np
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import torch
from torch.utils.data import TensorDataset, DataLoader
import joblib
from .landmark_normalization import normalize_landmarks
from .augmentation import augment_mirror_data

KEY_LANDMARK_INDICES = [
    # Right eye
    33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
    
    # Right iris
    468, 469, 470, 471, 472,

    # Left eye
    362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398,
    # Left iris
    473, 474, 475, 476, 477,

    # Right eyebrow
    # 107, 55, 108, #70, 46, 53, 52, 65, 55, 107, 66, 105, 63,

    # Left eyebrow
    # 336, 285, 337, #295, 282, 283, 276, 300, 293, 334, 296,

    # Right upper eyelid
    124, 113, 247, 30, 29, 27, 28, 56, 190, 189, 221, 222, 223, 224, 225,

    # Right lower eyelid 
    130, 226, 31, 228, 229, 230, 231, 232, 233, 245, 244, 112, 26, 22, 23, 24, 110, 25,243,

    #Left upper eyelid 
    413, 414, 286, 258, 257, 259, 260, 467, 342, 353, 445, 444, 443, 442, 441,

    # Left lower eyelid 
    464, 465, 453, 452, 451, 450, 449, 448, 261, 446, 359, 255, 339, 254, 253, 252, 256, 341,

    #Nose 
    1, 4, 5, 195, 197, 6, 168, 8, 9, #19, 196, 122, 188, 114, 217, 126, 209, 49, 48, 64, 237, 44, 35, 274, 309, 392, 294, 279, 429, 355, 437, 343, 412, 357, 351,

    #Chin 
    152,  175, 428, 199, 208#, 138#,135, 169, 170, 140, 171, 396, 369, 394, 364, 367

]

SYMMETRY_PAIRS = { #LEFT: RIGHT
    # Eyes
    33: 263, 7: 249, 163: 390, 144: 373, 145: 374, 153: 380, 154: 381, 155: 382,
    133: 362, 173: 398, 157: 384, 158: 385, 159: 386, 160: 387, 161: 388, 246: 466, #OK

    # Irises
    468: 473, 471: 474, 470: 475, 469: 476, 472: 477, #OK

    # Upper Eyelid
    124: 353, 113: 342, 247: 467, 30: 260, 29: 259, 27: 257, 28: 258, 56: 286,
    190: 414, 189: 413, 221: 441, 222: 442, 223: 443, 224: 444, 225: 445, #OK

    # Lower Eyelid
    130: 359, 226: 446, 31: 261, 228: 448, 229: 449, 230: 450, 231: 451, 232: 452,
    233: 453, 245: 465, 244: 464, 112: 341, 26: 256, 22: 252, 23: 253, 24: 254,
    110: 339, 25: 255,

    #Chin
    208:428


}

'''
H5 structure:
data/ --> main group
├── gaze_x --> dataset
├── gaze_y
├── is_valid
├── landmarks
├── left_eye
├── marker_x
├── marker_y
├── person_id
├── right_eye
├── source_csv  
└── timestamps
'''
def _prepare_h5_data(
    data_path,
    normalization_mode: str = "raw",
    grid_rows: int = 1,
    grid_cols: int = 5,
):

    with h5py.File(data_path, 'r') as f:
        data_group = f['data']
        
        all_landmarks = data_group['landmarks'][:]
        all_marker_x = data_group['marker_x'][:]
        all_marker_y = data_group['marker_y'][:]
        all_gaze_x = data_group['gaze_x'][:]
        all_gaze_y = data_group['gaze_y'][:]
        all_is_valid = data_group['is_valid'][:]
        all_person_ids = np.array([pid.decode('utf-8') for pid in data_group['person_id'][:]])
        all_source_csv = np.array([x.decode('utf-8') for x in data_group['source_csv'][:]])

    calibration_mask = np.isin(all_source_csv, ['data_3x3.csv', 'data_5x5.csv', 'data_smooth.csv'])
    all_landmarks = all_landmarks[calibration_mask]
    all_marker_x = all_marker_x[calibration_mask]
    all_marker_y = all_marker_y[calibration_mask]
    all_gaze_x = all_gaze_x[calibration_mask]
    all_gaze_y = all_gaze_y[calibration_mask]
    all_is_valid = all_is_valid[calibration_mask]
    all_person_ids = all_person_ids[calibration_mask]
    all_source_csv = all_source_csv[calibration_mask]

    SCREEN_W, SCREEN_H = 2560.0, 1440.0

    #validity check
    in_screen_mask = (all_gaze_x >= 0) & (all_gaze_x < SCREEN_W) & (all_gaze_y >= 0) & (all_gaze_y < SCREEN_H)
    valid_indices = np.where(all_is_valid & in_screen_mask)[0]
    print(f"Filtered out {np.sum(~(all_is_valid))} samples (invalid or off-screen)")

    landmarks_valid = all_landmarks[valid_indices]


    landmarks_normalized = normalize_landmarks(landmarks_valid,KEY_LANDMARK_INDICES, mode=normalization_mode)

    # TARGET SPECIFICATION:
    gaze_pixels = np.stack((all_gaze_x[valid_indices], all_gaze_y[valid_indices]), axis=-1)
    
    gx = np.clip(gaze_pixels[:, 0], 0, SCREEN_W - 1e-3)
    gy = np.clip(gaze_pixels[:, 1], 0, SCREEN_H - 1e-3)
    
    col_width = SCREEN_W / grid_cols
    row_height = SCREEN_H / grid_rows
    
    col_indices = (gx // col_width).astype(np.int64)
    row_indices = (gy // row_height).astype(np.int64)
    
    # Class ID = row * num_cols + col
    y = row_indices * grid_cols + col_indices

    # Keep pixel-space ground truth for evaluation/plots
    gaze_ground_truth = gaze_pixels
    person_ids_valid = all_person_ids[valid_indices]
    source_csv_valid = all_source_csv[valid_indices]
    
    X = landmarks_normalized.reshape(landmarks_normalized.shape[0], -1)

    return X, y, gaze_ground_truth, person_ids_valid, source_csv_valid


def _build_reference_frames(X_train_val, person_ids_train_val):
    """Compute per-person reference frames (mean feature vector)."""
    reference_frames = {}
    unique_person_ids_for_ref = np.unique(person_ids_train_val)
    for person_id in unique_person_ids_for_ref:
        person_mask = person_ids_train_val == person_id
        person_landmarks = X_train_val[person_mask]
        reference_frames[person_id] = person_landmarks.mean(axis=0)
    return reference_frames


def _apply_reference_frames(X, person_ids, reference_frames, global_mean=None):
    if global_mean is None:
        global_mean = X.mean(axis=0)
    X_delta = np.empty_like(X)
    for i in range(X.shape[0]):
        pid = person_ids[i]
        ref = reference_frames.get(pid, global_mean)
        X_delta[i] = X[i] - ref
    return X_delta


def get_h5_data_loaders(
    data_path,
    batch_size=32,
    train_person_ids=None,
    test_person_ids=None,
    normalization_mode: str = "raw",
    grid_rows: int = 3,
    grid_cols: int = 3,
):

    X, y, gaze_ground_truth, person_ids_valid, source_csv_valid = _prepare_h5_data(
        data_path=data_path,
        normalization_mode=normalization_mode,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
    )

    # split data into patients (train/test split by person)
    train_indices = np.where(np.isin(person_ids_valid, train_person_ids))[0]
    test_indices = np.where(np.isin(person_ids_valid, test_person_ids))[0]

    X_train_val, y_train_val = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]
    gaze_test = gaze_ground_truth[test_indices]
    
    # data needed to visualize each person later on 
    source_csv_test = source_csv_valid[test_indices]
    person_ids_test = person_ids_valid[test_indices]

    person_ids_train_val = person_ids_valid[train_indices]

    # --- Augmentation (Mirroring) ---

    print(f"Augmenting training data (mirroring)... Original size: {len(X_train_val)}")
    X_mirrored, y_mirrored = augment_mirror_data(
        X_train_val, 
        y_train_val, 
        grid_cols, 
        grid_rows, 
        KEY_LANDMARK_INDICES, 
        SYMMETRY_PAIRS
    )
    
    # X_train_val = np.concatenate([X_train_val, X_mirrored], axis=0)
    # y_train_val = np.concatenate([y_train_val, y_mirrored], axis=0)
    
    # Duplicate person_ids for the mirrored data so they stay in the same fold
    # person_ids_train_val = np.concatenate([person_ids_train_val, person_ids_train_val], axis=0)
    
    print(f"Augmentation done. New size: {len(X_train_val)}")
    # --------------------------------

    # Delta features (not used))
    reference_frames = _build_reference_frames(X_train_val, person_ids_train_val)

    X_train_val_delta = _apply_reference_frames(X_train_val, person_ids_train_val, reference_frames)
    global_mean = X_train_val.mean(axis=0)
    X_test_delta = _apply_reference_frames(X_test, person_ids_test, reference_frames, global_mean=global_mean)

    #X_train_val = X_train_val_delta
    #X_test = X_test_delta

    gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    train_idx, val_idx = next(gss.split(X_train_val, y_train_val, groups=person_ids_train_val))
    X_train_raw, y_train_raw = X_train_val[train_idx], y_train_val[train_idx]
    X_val_raw, y_val_raw     = X_train_val[val_idx], y_train_val[val_idx]

    # Scale features
    scaler = RobustScaler() #switched from standard to robust scaler to reduce outlier impact
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_val_scaled = scaler.transform(X_val_raw)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(scaler, 'scaler.pkl')
    # print("Scaler for X saved to scaler.pkl")

    y_train_scaled = y_train_raw #for classification, no scaling
    y_val_scaled = y_val_raw
    y_test_scaled = y_test

    y_train = torch.tensor(y_train_scaled, dtype=torch.long)
    y_val = torch.tensor(y_val_scaled, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.long)

    X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_val = torch.tensor(X_val_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

    unique_classes, class_counts = np.unique(y_train_scaled, return_counts=True)
    total_train_samples = len(y_train_scaled)
    print("Class distribution in training set:")
    for cls, cnt in zip(unique_classes, class_counts):
        print(f"  class {cls}: {cnt} samples ({cnt / total_train_samples * 100:.2f}%)")

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=unique_classes,
        y=y_train_scaled,
    )

    max_class = unique_classes.max()
    weight_tensor = torch.ones(max_class + 1, dtype=torch.float32)
    for cls, w in zip(unique_classes, class_weights):
        weight_tensor[int(cls)] = float(w)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=batch_size, shuffle=False)

    input_dim = X_train.shape[1]
    print(f"Train/Val dataset: {len(X_train)} training samples, {len(X_val)} validation samples")
    print(f"Test dataset: {len(X_test_tensor)} test samples")

    return train_loader, val_loader, test_loader, input_dim, source_csv_test, y_test, gaze_test, person_ids_test, weight_tensor

