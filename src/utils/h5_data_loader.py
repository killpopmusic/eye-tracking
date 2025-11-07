import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
import joblib

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
def get_h5_data_loaders(data_path, batch_size=32, num_persons=None):

    with h5py.File(data_path, 'r') as f:
        data_group = f['data']
        
        # Extract all data
        all_landmarks = data_group['landmarks'][:]
        all_marker_x = data_group['marker_x'][:]
        all_marker_y = data_group['marker_y'][:]
        all_is_valid = data_group['is_valid'][:]
        all_person_ids = np.array([pid.decode('utf-8') for pid in data_group['person_id'][:]])
        all_source_csv = np.array([x.decode('utf-8') for x in data_group['source_csv'][:]])

    # filter to get only 3x3 vs 5x5 for now 
    calibration_mask = np.isin(all_source_csv, ['data_3x3.csv', 'data_5x5.csv'])
    all_landmarks = all_landmarks[calibration_mask]
    all_marker_x = all_marker_x[calibration_mask]
    all_marker_y = all_marker_y[calibration_mask]
    all_is_valid = all_is_valid[calibration_mask]
    all_person_ids = all_person_ids[calibration_mask]
    print(f"Using only 3x3 and 5x5 calibration data: {np.sum(calibration_mask)} samples")

    # optional selection of number of patients
    person_ids_to_use = None
    if num_persons is not None:
        unique_person_ids = np.unique(all_person_ids)
        if num_persons > len(unique_person_ids):
            print(f"Warning: Requested {num_persons} persons, but only {len(unique_person_ids)} are available. Using all available persons.")
            num_persons = len(unique_person_ids)
        person_ids_to_use = unique_person_ids[:num_persons]
        print(f"Using data from the first {num_persons} persons: {person_ids_to_use}")

    if person_ids_to_use is not None:
        person_indices = np.where(np.isin(all_person_ids, person_ids_to_use))[0]
        all_landmarks = all_landmarks[person_indices]
        all_marker_x = all_marker_x[person_indices]
        all_marker_y = all_marker_y[person_indices]
        all_is_valid = all_is_valid[person_indices]
        all_person_ids = all_person_ids[person_indices]
    else:
        print("Using data from all available persons.")

    # Use only valid samples
    valid_indices = np.where(all_is_valid)[0]
    landmarks = all_landmarks[valid_indices]
    
    y = np.stack((all_marker_x[valid_indices], all_marker_y[valid_indices]), axis=-1)
    X = landmarks.reshape(landmarks.shape[0], -1)

    # data split
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # 
    # Initialize and fit the scaler 
    '''
    z=(x-mean)/std
    '''
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_val_scaled = scaler.transform(X_val_raw)

    joblib.dump(scaler, 'scaler.pkl')
    print("Scaler saved to scaler.pkl")

    # To tensors
    X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_val = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True) #--> returns (X_train[i], y_train[i])
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    input_dim = X_train.shape[1]
    print(f"Final dataset: {len(X_train)} training samples, {len(X_val)} validation samples")
    return train_loader, val_loader, input_dim
