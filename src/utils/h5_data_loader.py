import h5py
import numpy as np
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import RobustScaler, StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
import joblib
from .landmark_normalization import normalize_landmarks

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
def _prepare_h5_data_regression(
    data_path,
    normalization_mode: str = "raw",
    target_space: str = "normalized",
):

    KEY_LANDMARK_INDICES = [
    # Right eye
    33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
    
    # Right iris
    468, 469, 470, 471, 472,

    # Left eye
    362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398,
    # Left iris
    473, 474, 475, 476, 477,

    # Right upper eyelid
    124, 113, 247, 30, 29, 27, 28, 56, 190, 189, 221, 222, 223, 224, 225,

    # Right lower eyelid 
    130, 226, 31, 228, 229, 230, 231, 232, 233, 245, 244, 112, 26, 22, 23, 24, 110, 25,243,

    #Left upper eyelid 
    413, 414, 286, 258, 257, 259, 260, 467, 342, 353, 445, 444, 443, 442, 441,

    # Left lower eyelid 
    464, 465, 453, 452, 451, 450, 449, 448, 261, 446, 359, 255, 339, 254, 253, 252, 256, 341,

    #Nose 
    1, 4, 5, 195, 197, 6, 168, 8, 9, 

    #Chin 
    152,  175, 428, 199, 208

]
    print(f"Length of KEY_LANDMARK_INDICES: {len(KEY_LANDMARK_INDICES)}")
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

    # Filter out calibration data
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

    # TARGET SPECIFICATION (Regression):
    gaze_pixels = np.stack((all_gaze_x[valid_indices], all_gaze_y[valid_indices]), axis=-1)
    
    gx = np.clip(gaze_pixels[:, 0], 0, SCREEN_W - 1e-3)
    gy = np.clip(gaze_pixels[:, 1], 0, SCREEN_H - 1e-3)
    
    if target_space == "normalized":
        y_reg = np.stack((gx / SCREEN_W, gy / SCREEN_H), axis=-1).astype(np.float32)
    else:
        y_reg = np.stack((gx, gy), axis=-1).astype(np.float32)

    # Keep pixel-space ground truth for evaluation/plots
    gaze_ground_truth = gaze_pixels
    person_ids_valid = all_person_ids[valid_indices]
    source_csv_valid = all_source_csv[valid_indices]
    
    X = landmarks_normalized.reshape(landmarks_normalized.shape[0], -1)

    return X, y_reg, gaze_ground_truth, person_ids_valid, source_csv_valid

def get_h5_data_loaders_regression(
    data_path,
    batch_size=32,
    train_person_ids=None,
    test_person_ids=None,
    normalization_mode: str = "raw",
    target_space: str = "normalized",
    mode: str = "loso",
    scaler_path: str = "scaler.pkl",
):

    X, y_reg, gaze_ground_truth, person_ids_valid, source_csv_valid = _prepare_h5_data_regression(
        data_path=data_path,
        normalization_mode=normalization_mode,
        target_space=target_space,
    )

    # split data into subjects (train/test split by person)
    train_indices = np.where(np.isin(person_ids_valid, train_person_ids))[0]
    test_indices = np.where(np.isin(person_ids_valid, test_person_ids))[0]

    X_train_val, y_train_val = X[train_indices], y_reg[train_indices]
    X_test, y_test = X[test_indices], y_reg[test_indices]
    gaze_test = gaze_ground_truth[test_indices]
    
    # data needed to visualize each person later on 
    source_csv_test = source_csv_valid[test_indices]
    person_ids_test = person_ids_valid[test_indices]

    person_ids_train_val = person_ids_valid[train_indices]


    if mode == 'final':
        gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
        train_idx, val_idx = next(gss.split(X_train_val, y_train_val, groups=person_ids_train_val))
        X_train_raw, y_train_raw = X_train_val[train_idx], y_train_val[train_idx]
        X_val_raw, y_val_raw     = X_train_val[val_idx], y_train_val[val_idx]

       # X_train_raw, X_val_raw, y_train_raw, y_val_raw = train_test_split(
       #     X_train_val, y_train_val, test_size=0.1, random_state=42, stratify=y_train_val
       # )
    else:
        # LOSO
        gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
        train_idx, val_idx = next(gss.split(X_train_val, y_train_val, groups=person_ids_train_val))
        X_train_raw, y_train_raw = X_train_val[train_idx], y_train_val[train_idx]
        X_val_raw, y_val_raw     = X_train_val[val_idx], y_train_val[val_idx]

    # Scale features (X only)
    scaler = StandardScaler() #switched from standard to robust scaler to reduce outlier impact
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_val_scaled = scaler.transform(X_val_raw)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(scaler, scaler_path)
    print(f"Scaler for X saved to {scaler_path}")

    # Targets remain unscaled (normalized or pixel space)
    y_train = torch.tensor(y_train_raw, dtype=torch.float32)
    y_val = torch.tensor(y_val_raw, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_val = torch.tensor(X_val_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=batch_size, shuffle=False)

    input_dim = X_train.shape[1]
    print(f"Train/Val dataset: {len(X_train)} training samples, {len(X_val)} validation samples")
    print(f"Test dataset: {len(X_test_tensor)} test samples")

    return train_loader, val_loader, test_loader, input_dim, source_csv_test, y_test, gaze_test, person_ids_test

def split_calibration_data(X, y, source_csv, calibration_files=['data_3x3.csv'], calibration_fraction=0.15):
    is_calib_file = np.isin(source_csv, calibration_files)
    
    calib_indices = np.where(is_calib_file)[0]
    non_calib_indices = np.where(~is_calib_file)[0]
    
    if len(calib_indices) > 0:
        try:
            train_calib_idx, test_calib_idx = train_test_split(
                calib_indices, 
                train_size=calibration_fraction, 
                stratify=y[calib_indices],
                random_state=42
            )
        except ValueError:
            train_calib_idx, test_calib_idx = train_test_split(
                calib_indices, 
                train_size=calibration_fraction, 
                random_state=42
            )
        
        X_calib = X[train_calib_idx]
        y_calib = y[train_calib_idx]
        
        # Test set = rest of dataa
        final_test_indices = np.concatenate([test_calib_idx, non_calib_indices])
        final_test_indices.sort() 
 
        test_mask = np.zeros(len(X), dtype=bool)
        test_mask[final_test_indices] = True
        
        X_test = X[final_test_indices]
        y_test = y[final_test_indices]
        
        return X_calib, y_calib, X_test, y_test, test_mask
        
    else:
        return X[[]], y[[]], X, y, np.ones(len(X), dtype=bool)