import h5py
import numpy as np
import cv2
import math
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.utils.class_weight import compute_class_weight
import torch
from torch.utils.data import TensorDataset, DataLoader

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


def get_h5_data_loaders(
    data_path,
    batch_size: int = 32,
    train_person_ids=None,
    test_person_ids=None,
    grid_rows: int = 3,
    grid_cols: int = 3,
    frame_width: int = 2560,
    frame_height: int = 1440,
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

        # Right eyebrow
       # 70, 46, 53, 52, 65, 55, 107, 66, 105, 63,

        # Left eyebrow
       # 336, 285, 295, 282, 283, 276, 300, 293, 334, 296,

        # Right upper eyelid
        124, 113, 247, 30, 29, 27, 28, 56, 190, 189, 221, 222, 223, 224, 225,

        # Right lower eyelid 
        130, 226, 31, 228, 229, 230, 231, 232, 233, 245, 244, 112, 26, 22, 23, 24, 110, 25,

        #Left upper eyelid 
        413, 414, 286, 258, 257, 259, 260, 467, 342, 353, 445, 444, 443, 442, 441,

        # Left lower eyelid 
        464, 465, 453, 452, 451, 450, 449, 448, 261, 446, 359, 255, 339, 254, 253, 252, 256, 341,

        #Nose 
        19, 1, 4, 5, 195, 197, 6, 196,# 122, 188, 114, 217, 126, 209, 49, 48, 64, 237, 44, 35, 274, 309, 392, 294, 279, 429, 355, 437, 343, 412, 357, 351,

        #Chin 
        152,  175, 428, 199, 208, 138#,135, 169, 170, 140, 171, 396, 369, 394, 364, 367

    ]
    with h5py.File(data_path, 'r') as f:
        g = f['data']
        all_landmarks = g['landmarks'][:]
        all_gaze_x = g['gaze_x'][:]
        all_gaze_y = g['gaze_y'][:]
        all_is_valid = g['is_valid'][:]
        all_person_ids = np.array([pid.decode('utf-8') for pid in g['person_id'][:]])
        all_source_csv = np.array([x.decode('utf-8') for x in g['source_csv'][:]])

    calibration_mask = np.isin(all_source_csv, ['data_3x3.csv', 'data_5x5.csv', 'data_smooth.csv'])

    all_landmarks = all_landmarks[calibration_mask]
    all_gaze_x = all_gaze_x[calibration_mask]
    all_gaze_y = all_gaze_y[calibration_mask]
    all_is_valid = all_is_valid[calibration_mask]
    all_person_ids = all_person_ids[calibration_mask]
    all_source_csv = all_source_csv[calibration_mask]


    #Check for off-screen gaze
    SCREEN_W, SCREEN_H = float(frame_width), float(frame_height)

    in_screen_mask = (all_gaze_x >= 0) & (all_gaze_x < SCREEN_W) & (all_gaze_y >= 0) & (all_gaze_y < SCREEN_H)
    valid_idx = np.where(all_is_valid & in_screen_mask)[0]

    print(f"Filtered out {np.sum(~(all_is_valid))} samples (invalid or off-screen)")

    landmarks_valid = all_landmarks[valid_idx]  
    raw_subset = landmarks_valid[:, KEY_LANDMARK_INDICES, :2] 


    # Input 1: Gaze Vector
    # Right eye
    r_eye_corners = landmarks_valid[:, [33, 133], :2]
    r_eye_center = np.mean(r_eye_corners, axis=1)
    r_iris_pts = landmarks_valid[:, [468, 469, 470, 471, 472], :2]
    r_iris_center = np.mean(r_iris_pts, axis=1)
    r_gaze_vector = r_iris_center - r_eye_center

    # Left eye
    l_eye_corners = landmarks_valid[:, [263, 463], :2]
    l_eye_center = np.mean(l_eye_corners, axis=1)
    l_iris_pts = landmarks_valid[:, [473, 474, 475, 476, 477], :2]
    l_iris_center = np.mean(l_iris_pts, axis=1)
    l_gaze_vector = l_iris_center - l_eye_center

    # Normalize vectors 
    r_gaze_vector_norm = r_gaze_vector/ np.linalg.norm(r_gaze_vector, axis=1, keepdims=True)
    l_gaze_vector_norm = l_gaze_vector/ np.linalg.norm(l_gaze_vector, axis=1, keepdims=True)

    gaze_vectors = np.hstack((r_gaze_vector_norm, l_gaze_vector_norm)) # Shape (N, 4)
    print(f"Example normalized gaze vector (right x,y, left x,y): {gaze_vectors[0]}")

    # Input 2: Iris center

    iris_centers = np.hstack((r_iris_center[:, np.newaxis], l_iris_center[:, np.newaxis]))

    # --- Head Pose Estimation ---
    # 3D Model points (Generic Face Model)
    face_coordination_in_real_world = np.array([
        [285, 528, 200], # Nose tip (1)
        [285, 371, 152], # Nose top (9)
        [197, 574, 128], # Mouth left (57)
        [173, 425, 108], # Left eye left corner (130)
        [360, 574, 128], # Mouth right (287)
        [391, 425, 108]  # Right eye right corner (359)
    ], dtype=np.float64)

    # Indices in all_landmarks (MediaPipe indices)
    # 1, 9, 57, 130, 287, 359
    hp_indices = [1, 9, 57, 130, 287, 359]
    
    # Assume a generic webcam resolution for PnP calculation (e.g. 640x480)
    # This is independent of screen resolution. It's just to establish a coordinate system for PnP.
    CAM_W, CAM_H = 640, 480
    focal_length = CAM_W
    cam_matrix = np.array([[focal_length, 0, CAM_W / 2],
                           [0, focal_length, CAM_H / 2],
                           [0, 0, 1]], dtype=np.float64)
    dist_matrix = np.zeros((4, 1), dtype=np.float64)


    hp_landmarks = all_landmarks[valid_idx][:, hp_indices, :2] # Shape (N, 6, 2)

    head_pose_angles = []
    
    for i in range(len(hp_landmarks)):
        # Denormalize to the assumed camera resolution
        image_points = hp_landmarks[i] * np.array([CAM_W, CAM_H])
        
        success, rotation_vec, transition_vec = cv2.solvePnP(
            face_coordination_in_real_world, 
            image_points, 
            cam_matrix, 
            dist_matrix,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        rotation_matrix, _ = cv2.Rodrigues(rotation_vec)
        
        # Calculate Euler angles
        # Pitch (x), Yaw (y), Roll (z)
        x = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        y = math.atan2(-rotation_matrix[2, 0], math.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2))
        z = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        
        angles = np.array([x, y, z]) # Radians
        head_pose_angles.append(angles)

    scaler_angles = StandardScaler()
    head_pose_angles = np.array(head_pose_angles, dtype=np.float32) # Shape (N, 3)
    head_pose_angles = scaler_angles.fit_transform(head_pose_angles)
    print(f"Example head pose angles (radians): {head_pose_angles[0]}")
    # ----------------------------

    X_flat = raw_subset.reshape(raw_subset.shape[0], -1).astype(np.float32)

    # X = np.hstack((X_flat, gaze_vectors.astype(np.float32)))
    # X = gaze_vectors.astype(np.float32)
    X = np.hstack((X_flat, gaze_vectors, head_pose_angles)).astype(np.float32)
    

    gaze_pixels = np.stack((all_gaze_x[valid_idx], all_gaze_y[valid_idx]), axis=-1)
    gx = gaze_pixels[:, 0]
    gy = gaze_pixels[:, 1]

    col_w = SCREEN_W / grid_cols
    row_h = SCREEN_H / grid_rows
    col_indices = (gx // col_w).astype(np.int64)
    row_indices = (gy // row_h).astype(np.int64)
    y = row_indices * grid_cols + col_indices

    person_ids_valid = all_person_ids[valid_idx]
    source_csv_valid = all_source_csv[valid_idx]

    train_mask = np.isin(person_ids_valid, train_person_ids)
    test_mask = np.isin(person_ids_valid, test_person_ids)
    train_indices = np.where(train_mask)[0]
    test_indices = np.where(test_mask)[0]

    X_train_val, y_train_val = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]
    gaze_test = gaze_pixels[test_indices]
    source_csv_test = source_csv_valid[test_indices]
    person_ids_test = person_ids_valid[test_indices]

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42) #random people to val

    tr_idx, val_idx = next(gss.split(X_train_val, y_train_val, groups=person_ids_valid[train_indices]))
    X_train_raw, y_train_raw = X_train_val[tr_idx], y_train_val[tr_idx]
    X_val_raw, y_val_raw = X_train_val[val_idx], y_train_val[val_idx]

    y_train_t = torch.tensor(y_train_raw, dtype=torch.long)
    y_val_t = torch.tensor(y_val_raw, dtype=torch.long)
    y_test_t = torch.tensor(y_test, dtype=torch.long)
    X_train_t = torch.tensor(X_train_raw, dtype=torch.float32)
    X_val_t = torch.tensor(X_val_raw, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds = TensorDataset(X_val_t, y_val_t)
    test_ds = TensorDataset(X_test_t, y_test_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    input_dim = X_train_t.shape[1]
    print(f"Train/Val dataset: {len(X_train_t)} training samples, {len(X_val_t)} validation samples")
    print(f"Test dataset: {len(X_test_t)} test samples")

    unique_train, counts_train = np.unique(y_train_raw, return_counts=True)
    print(f"Train class distribution: {dict(zip(unique_train, counts_train))}")

    unique_test, counts_test = np.unique(y_test, return_counts=True)
    print(f"Test class distribution: {dict(zip(unique_test, counts_test))}")

    class_weights = compute_class_weight(
        class_weight='balanced', 
        classes=np.unique(y_train_raw), 
        y=y_train_raw
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

    return train_loader, val_loader, test_loader, input_dim, source_csv_test, y_test, gaze_test, person_ids_test, class_weights_tensor

