import numpy as np
import torch

def augment_mirror_data(
    X, 
    y, 
    grid_cols, 
    grid_rows, 
    current_landmarks, 
    symmetry_pairs
):

    is_torch = isinstance(X, torch.Tensor)
    if is_torch:
        X_np = X.cpu().numpy()
        y_np = y.cpu().numpy()
    else:
        X_np = X.copy()
        y_np = y.copy()

    N = X_np.shape[0]
    num_features = X_np.shape[1]
    num_landmarks = num_features // 2
    
    if len(current_landmarks) != num_landmarks:
        raise ValueError(f"Length of current_landmarks ({len(current_landmarks)}) "
                         f"does not match feature dimension ({num_landmarks} points).")

    # Reshape to (N, landmarks, 2)
    X_reshaped = X_np.reshape(N, num_landmarks, 2)
    
    # 1. Flip X coordinate
    # Assumes centered/normalized data where 0 is center.
    X_reshaped[:, :, 0] *= -1
    
    # 2. Swap Symmetric Landmarks
    # Build a lookup for the index of each landmark ID in the current dataset
    id_to_idx = {lid: i for i, lid in enumerate(current_landmarks)}
    
    # Normalize symmetry_pairs to a bidirectional dict
    pair_map = {}
    if isinstance(symmetry_pairs, (list, tuple)):
        for a, b in symmetry_pairs:
            pair_map[a] = b
            pair_map[b] = a
    elif isinstance(symmetry_pairs, dict):
        for a, b in symmetry_pairs.items():
            pair_map[a] = b
            pair_map[b] = a
            
    # Create permutation indices (default is identity)
    perm_indices = np.arange(num_landmarks)
    
    for i, lid in enumerate(current_landmarks):
        if lid in pair_map:
            partner_lid = pair_map[lid]
            if partner_lid in id_to_idx:
                # If the partner is also in our dataset, we swap to its position
                perm_indices[i] = id_to_idx[partner_lid]
            else:
                # If partner is missing from dataset, we can't swap correctly.
                # This implies the dataset is asymmetric. 
                # We leave it as is (no swap), which might be wrong but is the only safe option without dropping data.
                pass
                
    # Apply permutation
    X_mirrored = X_reshaped[:, perm_indices, :]
    
    # Flatten back
    X_mirrored = X_mirrored.reshape(N, -1)
    
    # 3. Update Labels (y)
    # y = row * grid_cols + col
    rows = y_np // grid_cols
    cols = y_np % grid_cols
    
    # Mirror column index
    new_cols = (grid_cols - 1) - cols
    y_mirrored = rows * grid_cols + new_cols
    
    if is_torch:
        return torch.from_numpy(X_mirrored).float(), torch.from_numpy(y_mirrored).long()
    else:
        return X_mirrored, y_mirrored
