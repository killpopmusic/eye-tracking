
import os
import json
from datetime import datetime
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_model(model, val_loader, model_path):
    os.makedirs('plot', exist_ok=True)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs('experiments', exist_ok=True)

    if model_path and os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location=next(model.parameters()).device)
            state_dict = checkpoint.get('model_state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
            model.load_state_dict(state_dict)
            print(f"Loaded model weights from {model_path}")
        except Exception as e:
            print(f"Could not load model from {model_path}. Using the model passed as an argument. Error: {e}")

    device = next(model.parameters()).device
    model.eval()

    all_labels = []
    all_preds = []

    with torch.inference_mode():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_preds.extend(outputs.cpu().numpy()) # back to cpu to get np arrays
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Calculate metrics
    mae = mean_absolute_error(all_labels, all_preds)
    rmse = np.sqrt(mean_squared_error(all_labels, all_preds))
    r2 = r2_score(all_labels, all_preds)
    l2_norm = np.mean(np.linalg.norm(all_labels - all_preds, axis=1))

    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R-squared (R²): {r2:.4f}")
    print(f"L2 Norm: {l2_norm:.4f}")

    # experiment data to json
    run_summary = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model_name": type(model).__name__,
        "metrics": {
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
            "l2_norm": float(l2_norm),
        },
    }
    run_file = os.path.join('experiments', f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json")
    try:
        with open(run_file, 'w') as f:
            json.dump(run_summary, f, indent=2)
        print(f"Run summary saved to {run_file}")
    except Exception as e:
        print(f"Failed to save run summary JSON: {e}")

    # visualization attempt
    plt.figure(figsize=(25, 14))
    plt.scatter(all_labels[:, 0], all_labels[:, 1], label='Actual Points', alpha=0.6, s=100)
    plt.scatter(all_preds[:, 0], all_preds[:, 1], label='Predicted Points', alpha=0.6, s=100, c='r', marker='x')
    
    for i in range(len(all_labels)):
        plt.plot([all_labels[i, 0], all_preds[i, 0]], [all_labels[i, 1], all_preds[i, 1]], 'k-', alpha=0.2)

    plt.title('Actual vs. Predicted Gaze Points')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 2560)
    plt.ylim(0, 1440)
    plt.gca().invert_yaxis() 

    plt.savefig('plot/prediction_visualization.png')
    print("Prediction visualization saved to plot/prediction_visualization.png")
