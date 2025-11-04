import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
import joblib # save scaler

def get_data_loaders(data_path, batch_size=32):

    with open(data_path) as f:
        data = json.load(f)

    #Data to dictionary
    records = []
    for grid_size, points in data.items():
        if grid_size in ['3x3', '5x5']:
            for point_data in points:
                if point_data['landmarks']:
                    records.append({
                        'grid_size': grid_size,
                        'point_x': point_data['point'][0],
                        'point_y': point_data['point'][1],
                        'distance': point_data['distance'],
                        'landmarks': np.array([list(landmark.values()) for landmark in point_data['landmarks']]).flatten()
                    })

    df = pd.DataFrame(records)

    #print(f"Example records: {df.sample(5)}")

    # Prepare data for PyTorch
    X = np.vstack(df['landmarks'].values)
    y = df[['point_x', 'point_y']].values

    X_train_raw, X_val_raw, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

    # Initialize and fit the scaler 
    '''
    z=(x-mean)/std
    '''
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    
    # Transform the validation data using the scaler
    X_val_scaled = scaler.transform(X_val_raw)

    # Scaler save
    joblib.dump(scaler, 'scaler.pkl')
    print("Scaler saved to scaler.pkl")

    # Data to tensors 
    X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_val = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, X_train.shape[1]
