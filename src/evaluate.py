
import os
import json
from datetime import datetime
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_model(model, test_loader, model_path, source_csv_test, y_test, gaze_test):
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
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Calculate metrics for model predictions
    mae = mean_absolute_error(all_labels, all_preds)
    rmse = np.sqrt(mean_squared_error(all_labels, all_preds))
    r2 = r2_score(all_labels, all_preds)
    l2_norm = np.mean(np.linalg.norm(all_labels - all_preds, axis=1))

    print(f"Test Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Test Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Test R-squared (R²): {r2:.4f}")
    print(f"Test L2 Norm: {l2_norm:.4f}")

    # Calculate metrics for ground truth vs markers
    gt_mae = mean_absolute_error(all_labels, gaze_test)
    gt_rmse = np.sqrt(mean_squared_error(all_labels, gaze_test))
    gt_r2 = r2_score(all_labels, gaze_test)
    gt_l2_norm = np.mean(np.linalg.norm(all_labels - gaze_test, axis=1))

    print("\n--- Ground Truth (Eyetracker) vs. Markers (Target) ---")
    print(f"GT Mean Absolute Error (MAE): {gt_mae:.4f}")
    print(f"GT Root Mean Squared Error (RMSE): {gt_rmse:.4f}")
    print(f"GT R-squared (R²): {gt_r2:.4f}")
    print(f"GT L2 Norm: {gt_l2_norm:.4f}")

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
        "ground_truth_metrics": {
            "mae": float(gt_mae),
            "rmse": float(gt_rmse),
            "r2": float(gt_r2),
            "l2_norm": float(gt_l2_norm),
        }
    }
    run_file = os.path.join('experiments', f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json")
    try:
        with open(run_file, 'w') as f:
            json.dump(run_summary, f, indent=2)
        print(f"Test run summary saved to {run_file}")
    except Exception as e:
        print(f"Failed to save test run summary JSON: {e}")

    # Visualization for 3x3, 5x5, and smooth data
    plot_test_visualizations(all_labels, all_preds, source_csv_test)

def plot_test_visualizations(all_labels, all_preds, source_csv_test):
    # 3x3 grid
    mask_3x3 = source_csv_test == 'data_3x3.csv'
    if np.any(mask_3x3):
        plt.figure(figsize=(12, 8))
        plt.scatter(all_labels[mask_3x3, 0], all_labels[mask_3x3, 1], label='Actual Points (3x3)', alpha=0.6, s=100)
        plt.scatter(all_preds[mask_3x3, 0], all_preds[mask_3x3, 1], label='Predicted Points (3x3)', alpha=0.6, s=100, c='r', marker='x')
        for i in np.where(mask_3x3)[0]:
            plt.plot([all_labels[i, 0], all_preds[i, 0]], [all_labels[i, 1], all_preds[i, 1]], 'k-', alpha=0.2)
        plt.title('Test Set: Actual vs. Predicted Gaze Points (3x3)')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.legend()
        plt.grid(True)
        plt.xlim(0, 2560)
        plt.ylim(0, 1440)
        plt.gca().invert_yaxis()
        plt.savefig('plot/test_prediction_3x3.png')
        print("Test prediction visualization for 3x3 saved to plot/test_prediction_3x3.png")

    # 5x5 grid
    mask_5x5 = source_csv_test == 'data_5x5.csv'
    if np.any(mask_5x5):
        plt.figure(figsize=(12, 8))
        plt.scatter(all_labels[mask_5x5, 0], all_labels[mask_5x5, 1], label='Actual Points (5x5)', alpha=0.6, s=100)
        plt.scatter(all_preds[mask_5x5, 0], all_preds[mask_5x5, 1], label='Predicted Points (5x5)', alpha=0.6, s=100, c='r', marker='x')
        for i in np.where(mask_5x5)[0]:
            plt.plot([all_labels[i, 0], all_preds[i, 0]], [all_labels[i, 1], all_preds[i, 1]], 'k-', alpha=0.2)
        plt.title('Test Set: Actual vs. Predicted Gaze Points (5x5)')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.legend()
        plt.grid(True)
        plt.xlim(0, 2560)
        plt.ylim(0, 1440)
        plt.gca().invert_yaxis()
        plt.savefig('plot/test_prediction_5x5.png')
        print("Test prediction visualization for 5x5 saved to plot/test_prediction_5x5.png")

    # Smooth pursuit
    mask_smooth = source_csv_test == 'data_smooth.csv'
    if np.any(mask_smooth):
        plt.figure(figsize=(12, 8))
        plt.plot(all_labels[mask_smooth, 0], all_labels[mask_smooth, 1], 'b-o', label='Actual Trajectory', markersize=5)
        plt.plot(all_preds[mask_smooth, 0], all_preds[mask_smooth, 1], 'r-x', label='Predicted Trajectory', markersize=5)
        plt.title('Test Set: Actual vs. Predicted Gaze Trajectory (Smooth Pursuit)')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.legend()
        plt.grid(True)
        plt.xlim(0, 2560)
        plt.ylim(0, 1440)
        plt.gca().invert_yaxis()
        plt.savefig('plot/test_prediction_smooth.png')
        print("Test prediction visualization for smooth pursuit saved to plot/test_prediction_smooth.png")

