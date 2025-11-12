
import os
import json
from datetime import datetime
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

def evaluate_model(model, test_loader, model_path, source_csv_test, y_test, gaze_test, person_ids_test, hyperparameters):
    os.makedirs('plot', exist_ok=True)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs('experiments', exist_ok=True)

    # Load scalers
    try:
        y_scaler = joblib.load('y_scaler.pkl')
        print("Loaded y_scaler.pkl for inverse transformation.")
    except FileNotFoundError:
        print("Warning: y_scaler.pkl not found. Predictions and labels will not be de-normalized.")
        y_scaler = None

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

    all_labels_scaled = []
    all_preds_scaled = []

    with torch.inference_mode():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_preds_scaled.extend(outputs.cpu().numpy())
            all_labels_scaled.extend(labels.cpu().numpy())

    all_preds_scaled = np.array(all_preds_scaled)
    all_labels_scaled = np.array(all_labels_scaled)

    # De-normalize predictions and labels if scaler is available
    if y_scaler:
        all_preds = y_scaler.inverse_transform(all_preds_scaled)
        # The original y_test is already in the original scale
        all_labels = y_test
    else:
        all_preds = all_preds_scaled
        all_labels = all_labels_scaled

    unique_person_ids = np.unique(person_ids_test)

    for person_id in unique_person_ids:
        person_mask = person_ids_test == person_id
        person_labels = all_labels[person_mask]
        person_preds = all_preds[person_mask]
        person_gaze = gaze_test[person_mask]
        person_source_csv = source_csv_test[person_mask]

        print(f"\n--- Evaluating for Person: {person_id} ---")

        # Calculate metrics for model predictions
        mae = mean_absolute_error(person_labels, person_preds)
        rmse = np.sqrt(mean_squared_error(person_labels, person_preds))
        r2 = r2_score(person_labels, person_preds)
        l2_norm = np.mean(np.linalg.norm(person_labels - person_preds, axis=1))

        print(f"Test Mean Absolute Error (MAE): {mae:.4f}")
        print(f"Test Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"Test R-squared (R²): {r2:.4f}")
        print(f"Test L2 Norm: {l2_norm:.4f}")

        # Calculate metrics for ground truth vs markers
        gt_mae = mean_absolute_error(person_labels, person_gaze)
        gt_rmse = np.sqrt(mean_squared_error(person_labels, person_gaze))
        gt_r2 = r2_score(person_labels, person_gaze)
        gt_l2_norm = np.mean(np.linalg.norm(person_labels - person_gaze, axis=1))

        print("\n--- Ground Truth (Eyetracker) vs. Markers (Target) ---")
        print(f"GT Mean Absolute Error (MAE): {gt_mae:.4f}")
        print(f"GT Root Mean Squared Error (RMSE): {gt_rmse:.4f}")
        print(f"GT R-squared (R²): {gt_r2:.4f}")
        print(f"GT L2 Norm: {gt_l2_norm:.4f}")

        # experiment data to json
        run_summary = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "model_name": type(model).__name__,
            "person_id": person_id,
            "hyperparameters": hyperparameters,
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
        run_file = os.path.join('experiments', f"run_{person_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json")
        try:
            with open(run_file, 'w') as f:
                json.dump(run_summary, f, indent=2)
            print(f"Test run summary for {person_id} saved to {run_file}")
        except Exception as e:
            print(f"Failed to save test run summary JSON for {person_id}: {e}")

        # Visualization for 3x3, 5x5, and smooth data
        plot_test_visualizations(person_labels, person_preds, person_source_csv, person_id)

def plot_test_visualizations(all_labels, all_preds, source_csv_test, person_id):
    plot_dir = f'plot/{person_id}'
    os.makedirs(plot_dir, exist_ok=True)
    
    # 3x3 grid
    mask_3x3 = source_csv_test == 'data_3x3.csv'
    if np.any(mask_3x3):
        plt.figure(figsize=(12, 8))
        plt.scatter(all_labels[mask_3x3, 0], all_labels[mask_3x3, 1], label='Actual Points (3x3)', alpha=0.6, s=100)
        plt.scatter(all_preds[mask_3x3, 0], all_preds[mask_3x3, 1], label='Predicted Points (3x3)', alpha=0.6, s=100, c='r', marker='x')
        for i in np.where(mask_3x3)[0]:
            plt.plot([all_labels[i, 0], all_preds[i, 0]], [all_labels[i, 1], all_preds[i, 1]], 'k-', alpha=0.2)
        plt.title(f'Test Set: Actual vs. Predicted Gaze Points (3x3) - Person {person_id}')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.legend()
        plt.grid(True)
        plt.xlim(0, 2560)
        plt.ylim(0, 1440)
        plt.gca().invert_yaxis()
        plt.savefig(os.path.join(plot_dir, 'test_prediction_3x3.png'))
        print(f"Test prediction visualization for 3x3 for person {person_id} saved to {os.path.join(plot_dir, 'test_prediction_3x3.png')}")
        plt.close()

    # 5x5 grid
    mask_5x5 = source_csv_test == 'data_5x5.csv'
    if np.any(mask_5x5):
        plt.figure(figsize=(12, 8))
        plt.scatter(all_labels[mask_5x5, 0], all_labels[mask_5x5, 1], label='Actual Points (5x5)', alpha=0.6, s=100)
        plt.scatter(all_preds[mask_5x5, 0], all_preds[mask_5x5, 1], label='Predicted Points (5x5)', alpha=0.6, s=100, c='r', marker='x')
        for i in np.where(mask_5x5)[0]:
            plt.plot([all_labels[i, 0], all_preds[i, 0]], [all_labels[i, 1], all_preds[i, 1]], 'k-', alpha=0.2)
        plt.title(f'Test Set: Actual vs. Predicted Gaze Points (5x5) - Person {person_id}')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.legend()
        plt.grid(True)
        plt.xlim(0, 2560)
        plt.ylim(0, 1440)
        plt.gca().invert_yaxis()
        plt.savefig(os.path.join(plot_dir, 'test_prediction_5x5.png'))
        print(f"Test prediction visualization for 5x5 for person {person_id} saved to {os.path.join(plot_dir, 'test_prediction_5x5.png')}")
        plt.close()

    # Smooth pursuit
    mask_smooth = source_csv_test == 'data_smooth.csv'
    if np.any(mask_smooth):
        plt.figure(figsize=(12, 8))
        plt.plot(all_labels[mask_smooth, 0], all_labels[mask_smooth, 1], 'b-o', label='Actual Trajectory', markersize=5)
        plt.plot(all_preds[mask_smooth, 0], all_preds[mask_smooth, 1], 'r-x', label='Predicted Trajectory', markersize=5)
        plt.title(f'Test Set: Actual vs. Predicted Gaze Trajectory (Smooth Pursuit) - Person {person_id}')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.legend()
        plt.grid(True)
        plt.xlim(0, 2560)
        plt.ylim(0, 1440)
        plt.gca().invert_yaxis()
        plt.savefig(os.path.join(plot_dir, 'test_prediction_smooth.png'))
        print(f"Test prediction visualization for smooth pursuit for person {person_id} saved to {os.path.join(plot_dir, 'test_prediction_smooth.png')}")
        plt.close()


