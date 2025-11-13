import numpy as np

def normalize_landmarks(landmarks, key_indices=None):

    stable_indices = [1, 6, 152, 234, 454] 

    landmarks_xy = landmarks[:, :, :2]
    
    # Mean head center
    head_center = landmarks_xy[:, stable_indices, :].mean(axis=1, keepdims=True)
    
    # Center all landmarks by subtracting the calculated head center
    centered_landmarks = landmarks_xy - head_center
    
    # Calculate the scale 
    scale = np.linalg.norm(centered_landmarks, axis=2).mean(axis=1, keepdims=True)
    
    # Avoid division by zero in case of malformed data
    scale[scale < 1e-8] = 1e-8
    
    # Scale the landmarks 
    normalized_landmarks = centered_landmarks / scale[:, :, np.newaxis]
    
    print("Landmarks normalized relative to head position and scale.")

    if key_indices is not None:
        print(f"Returning {len(key_indices)} selected key landmarks.")
        return normalized_landmarks[:, key_indices, :]
    
    return normalized_landmarks
