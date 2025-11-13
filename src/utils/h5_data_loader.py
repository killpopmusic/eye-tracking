import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
def get_h5_data_loaders(data_path, batch_size=32, train_person_ids=None, test_person_ids=None):

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
        70, 46, 53, 52, 65, 55, 107, 66, 105, 63,

        # Left eyebrow
        336, 285, 295, 282, 283, 276, 300, 293, 334, 296,

        # Right upper eyelid
        124, 113, 247, 30, 29, 27, 28, 56, 190, 189, 221, 222, 223, 224, 225,

        # Right lower eyelid 
        130, 226, 31, 228, 229, 230, 231, 232, 233, 245, 244, 112, 26, 22, 23, 24, 110, 25,

        #Left upper eyelid 
        413, 414, 286, 258, 257, 259, 260, 467, 342, 353, 445, 444, 443, 442, 441,

        # Left lower eyelid 
        464, 465, 453, 452, 451, 450, 449, 448, 261, 446, 359, 255, 339, 254, 253, 252, 256, 341

    ]

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

    # Use only valid samples
    valid_indices = np.where(all_is_valid)[0]
    landmarks_valid = all_landmarks[valid_indices]

    landmarks_normalized = normalize_landmarks(landmarks_valid, KEY_LANDMARK_INDICES)

    y = np.stack((all_marker_x[valid_indices], all_marker_y[valid_indices]), axis=-1)
    gaze_ground_truth = np.stack((all_gaze_x[valid_indices], all_gaze_y[valid_indices]), axis=-1)
    person_ids_valid = all_person_ids[valid_indices]
    source_csv_valid = all_source_csv[valid_indices]
    
    X = landmarks_normalized.reshape(landmarks_normalized.shape[0], -1)

    # split data into patients
    train_indices = np.where(np.isin(person_ids_valid, train_person_ids))[0]
    test_indices = np.where(np.isin(person_ids_valid, test_person_ids))[0]

    X_train_val, y_train_val = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]
    gaze_test = gaze_ground_truth[test_indices]
    
    # data needed to visualize each person later on 
    source_csv_test = source_csv_valid[test_indices]
    person_ids_test = person_ids_valid[test_indices]

    # train/val split
    X_train_raw, X_val_raw, y_train_raw, y_val_raw = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_val_scaled = scaler.transform(X_val_raw)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, 'scaler.pkl')
    print("Scaler for X saved to scaler.pkl")

    # Scale targets
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train_raw)
    y_val_scaled = y_scaler.transform(y_val_raw)
    y_test_scaled = y_scaler.transform(y_test)
    joblib.dump(y_scaler, 'y_scaler.pkl')
    print("Scaler for y saved to y_scaler.pkl")

    X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train = torch.tensor(y_train_scaled, dtype=torch.float32)
    X_val = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_val = torch.tensor(y_val_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=batch_size, shuffle=False)

    input_dim = X_train.shape[1]
    print(f"Train/Val dataset: {len(X_train)} training samples, {len(X_val)} validation samples")
    print(f"Test dataset: {len(X_test_tensor)} test samples")
    
    return train_loader, val_loader, test_loader, input_dim, source_csv_test, y_test, gaze_test, person_ids_test

