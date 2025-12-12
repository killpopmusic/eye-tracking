import argparse
import os
import json
from datetime import datetime

import torch
import h5py
import numpy as np

from src.train import train_regressor
from src.evaluate import evaluate_regressor
from src.utils.h5_data_loader import get_h5_data_loaders_regression, _prepare_h5_data_regression
from src.models.gaze_regressor import GazeRegressor
from src.models.gaze_regressor_min import GazeRegressorMin


def main():
    parser = argparse.ArgumentParser(description="Gaze Regression Model Training and Evaluation")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate', 'train_final'], help='Mode to run the script in.')
    parser.add_argument('--model_name', type=str, default='GazeRegressor', help='Name of the model class to use.')
    parser.add_argument('--data_path', type=str, default='data/HybridGaze.h5', help='Path to the training/evaluation data.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training and validation.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate for the optimizer.')
    parser.add_argument('--model_path', type=str, default='models/gaze_regressor.pth', help='Path to save or load the model.')
    parser.add_argument('--target_space', type=str, default='normalized', choices=['normalized', 'pixel'], help='Target space for regression labels.')

    args = parser.parse_args()

    output_dim = 2

    model_map = {
        'GazeRegressor': GazeRegressor,
        'GazeRegressorMin': GazeRegressorMin,
    }
    model_class = model_map.get(args.model_name)
    
    if not model_class:
        print(f"Model {args.model_name} not found.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"

    X_all, y_all, gaze_all, person_ids_all, source_csv_all = _prepare_h5_data_regression(
        data_path=args.data_path,
        normalization_mode="raw",
        target_space=args.target_space,
    )

    all_person_ids = np.unique(person_ids_all)

    hyperparameters = {
        "epochs": args.epochs, 
        "batch_size": args.batch_size, 
        "learning_rate": args.lr
    }

    if args.mode == 'train_final':
        print("\n===== Final Training on ALL Subjects =====")

        train_person_ids = all_person_ids
        test_person_ids = np.array([]) # No test subjects

        os.makedirs('trained_models', exist_ok=True)
        scaler_save_path = os.path.join('trained_models', 'production_scaler.pkl')

        data_loader_params = {
            'data_path': args.data_path,
            'batch_size': args.batch_size,
            'train_person_ids': train_person_ids,
            'test_person_ids': test_person_ids,
            'normalization_mode': 'raw',
            'target_space': args.target_space,
            'mode': 'final',
            'scaler_path': scaler_save_path
        }
        
        (
            train_loader,
            val_loader,
            _, # test_loader is empty
            input_features,
            _, _, _, _, # unused test data
            class_weights,
        ) = get_h5_data_loaders_regression(**data_loader_params)

        model = model_class(input_features, output_dim=output_dim).to(device)
        
        # Save to a distinct final path
        final_model_path = os.path.join('trained_models', 'final_regressor.pth')

        train_regressor(
            model,
            train_loader,
            val_loader,
            num_epochs=args.epochs,
            lr=args.lr,
            model_path=final_model_path,
            patience=args.epochs #  disable early stopping
        )
        print(f"Final model saved to {final_model_path}")
        print(f"Scaler saved to {scaler_save_path}")
        return

    all_person_results = []

    for leave_out_pid in all_person_ids:
        print(f"\n===== LOSO fold: test person {leave_out_pid} =====")

        train_person_ids = all_person_ids[all_person_ids != leave_out_pid]
        test_person_ids = np.array([leave_out_pid])

        data_loader_params = {
            'data_path': args.data_path,
            'batch_size': args.batch_size,
            'train_person_ids': train_person_ids,
            'test_person_ids': test_person_ids,
            'normalization_mode': 'raw',
            'target_space': args.target_space,
        }
        
        (
            train_loader,
            val_loader,
            test_loader,
            input_features,
            source_csv_test,
            y_test,
            gaze_test,
            person_ids_test
        ) = get_h5_data_loaders_regression(**data_loader_params)

        model = model_class(input_features, output_dim=output_dim).to(device)

        fold_model_path = args.model_path
        base, ext = os.path.splitext(args.model_path)
        fold_model_path = f"{base}_person_{leave_out_pid}{ext}"

        if args.mode == 'train':
            train_regressor(
                model,
                train_loader,
                val_loader,
                num_epochs=args.epochs,
                lr=args.lr,
                model_path=fold_model_path,
            )
            print("Regressor training finished. Starting evaluation on the test set...")

        if args.mode in ['train', 'evaluate']:
            if not os.path.exists(fold_model_path):
                print(f"Model file not found at {fold_model_path}, skipping this fold.")
                continue

            fold_results = evaluate_regressor(
                model,
                test_loader,
                fold_model_path,
                person_ids_test,
                source_csv_test,
                hyperparameters,
                target_space=args.target_space,
            )

            all_person_results.extend(fold_results)

    excluded_for_aggregate =  ['2025_06_02_11_09_16', '2025_05_27_10_57_49', '2025_06_07_22_33_55']

    included_for_aggregate = [
        r for r in all_person_results if r["person_id"] not in excluded_for_aggregate
    ]

    if included_for_aggregate:
        maes = [r["metrics"]["mae_px"] for r in included_for_aggregate]
        rmses = [r["metrics"]["rmse_px"] for r in included_for_aggregate]
        r2s = [r["metrics"].get("r2") for r in included_for_aggregate if r["metrics"].get("r2") is not None]
        within3 = [r["metrics"].get("within_3deg_pct", 0.0) for r in included_for_aggregate]

        aggregate_metrics = {
            "mae_px_mean": float(np.mean(maes)),
            "mae_px_std": float(np.std(maes)),
            "rmse_px_mean": float(np.mean(rmses)),
            "rmse_px_std": float(np.std(rmses)),
            "r2_mean": float(np.mean(r2s)) if r2s else None,
            "r2_std": float(np.std(r2s)) if r2s else None,
            "within_3deg_pct_mean": float(np.mean(within3)),
            "within_3deg_pct_std": float(np.std(within3)),
            "included_person_ids": [r["person_id"] for r in included_for_aggregate],
            "excluded_person_ids": list(excluded_for_aggregate),
        }
    else:
        aggregate_metrics = None

    summary = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model_name": model_class.__name__,
        "model_path_template": args.model_path,
        "hyperparameters": hyperparameters,
        "task": "regression",
        "target_space": args.target_space,
        "per_person_results": all_person_results,
        "aggregate_metrics": aggregate_metrics,
    }

    os.makedirs("experiments", exist_ok=True)
    summary_path = os.path.join(
        "experiments",
        f"run_regressor_loso_summary_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json",
    )

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Global LOSO summary saved to {summary_path}")

if __name__ == '__main__':
    main()


