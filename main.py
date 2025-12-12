import argparse
import os
import json
from datetime import datetime
import torch
import h5py
import numpy as np

from src.train import train_classifier, fine_tune_model
from src.evaluate import evaluate_classifier
from src.utils.h5_data_loader import get_h5_data_loaders, _prepare_h5_data, split_calibration_data
from src.models.gaze_classifier import GazeClassifier
from src.models.gaze_res_mlp import GazeResMLP
from src.models.gaze_classifier_min import GazeClassifierMin

def main():
    parser = argparse.ArgumentParser(description="Gaze Classification Model Training and Evaluation")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate', 'train_final'], help='Mode to run the script in.')
    parser.add_argument('--model_name', type=str, default='GazeClassifier', help='Name of the model class to use.')
    parser.add_argument('--data_path', type=str, default='data/HybridGaze.h5', help='Path to the training/evaluation data.')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training and validation.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate for the optimizer.')
    parser.add_argument('--model_path', type=str, default='models/gaze_classifier.pth', help='Path to save or load the model.')
    parser.add_argument('--grid_rows', type=int, default=3, help='Number of rows in the classification grid.')
    parser.add_argument('--grid_cols', type=int, default=3, help='Number of columns in the classification grid.')
    parser.add_argument('--calibrate', action='store_true', help='Enable calibration for the test person using 3x3 grid data.')
    parser.add_argument('--eval_exclude_3x3', action='store_true', help='Exclude 3x3 data from evaluation (for fair comparison).')

    args = parser.parse_args()

    num_classes = args.grid_rows * args.grid_cols

    model_map = {
        'GazeClassifier': GazeClassifier,
        'GazeResMLP': GazeResMLP,
        'GazeClassifierMin': GazeClassifierMin,
    }
    model_class = model_map.get(args.model_name)
    
    if not model_class:
        print(f"Model {args.model_name} not found.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"

    X_all, y_all, gaze_all, person_ids_all, source_csv_all = _prepare_h5_data(data_path=args.data_path, normalization_mode="raw", grid_rows=args.grid_rows, grid_cols=args.grid_cols,)

    all_person_ids = np.unique(person_ids_all)

    hyperparameters = {"epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.lr, "grid_size": f"{args.grid_rows}x{args.grid_cols}"}

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
            'grid_rows': args.grid_rows,
            'grid_cols': args.grid_cols,
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
        ) = get_h5_data_loaders(**data_loader_params)

        model = model_class(input_features, num_classes=num_classes).to(device)
        
        # Save to a distinct final path
        final_model_path = os.path.join('trained_models', 'final_model.pth')

        train_classifier(
            model,
            train_loader,
            val_loader,
            num_epochs=args.epochs,
            lr=args.lr,
            model_path=final_model_path,
            class_weights=class_weights,
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
            'grid_rows': args.grid_rows,
            'grid_cols': args.grid_cols
        }
        
        (
            train_loader,
            val_loader,
            test_loader,
            input_features,
            source_csv_test,
            y_test,
            gaze_test,
            person_ids_test,
            class_weights,
        ) = get_h5_data_loaders(**data_loader_params)

        model = model_class(input_features, num_classes=num_classes).to(device)

        fold_model_path = args.model_path
        base, ext = os.path.splitext(args.model_path)
        fold_model_path = f"{base}_person_{leave_out_pid}{ext}"

        if args.mode == 'train':
            train_classifier(
                model,
                train_loader,
                val_loader,
                num_epochs=args.epochs,
                lr=args.lr,
                model_path=fold_model_path,
                class_weights=class_weights,
            )
            print("Classifier training finished. Starting evaluation on the test set...")

        if args.mode in ['train', 'evaluate']:
            if not os.path.exists(fold_model_path):
                print(f"Model file not found at {fold_model_path}, skipping this fold.")
                continue

            try:
                checkpoint = torch.load(fold_model_path, map_location=device)
                state_dict = checkpoint.get('model_state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
                model.load_state_dict(state_dict)
            except Exception as e:
                print(f"Error loading model: {e}")

            calib_mask = None

            if args.calibrate or args.eval_exclude_3x3:
                X_test_all = test_loader.dataset.tensors[0]
                y_test_all = test_loader.dataset.tensors[1]

                X_calib, y_calib, X_eval, y_eval, calib_mask = split_calibration_data(
                    X_test_all,
                    y_test_all,
                    source_csv_test,
                    calibration_files=['data_3x3.csv'],
                    calibration_fraction=1.0,
                )

            if args.calibrate:
                print(f"--- Calibrating for person {leave_out_pid} ---")
                if len(X_calib) > 0:
                    calib_loader = torch.utils.data.DataLoader(
                        torch.utils.data.TensorDataset(X_calib, y_calib),
                        batch_size=16,
                        shuffle=True,
                    )
                    fine_tune_model(model, calib_loader, epochs=10, lr=0.00005)
                else:
                    print("No calibration data found (data_3x3.csv). Skipping calibration.")

            if args.calibrate or args.eval_exclude_3x3:
                if calib_mask is not None:
                    test_loader = torch.utils.data.DataLoader(
                        torch.utils.data.TensorDataset(X_eval, y_eval),
                        batch_size=args.batch_size,
                        shuffle=False,
                    )

                    eval_mask = ~calib_mask
                    source_csv_test = source_csv_test[eval_mask]
                    y_test = y_test[eval_mask]
                    gaze_test = gaze_test[eval_mask]
                    person_ids_test = person_ids_test[eval_mask]

            fold_results = evaluate_classifier(
                model,
                test_loader,
                None,
                source_csv_test,
                y_test,
                gaze_test,
                person_ids_test,
                hyperparameters,
                grid_rows=args.grid_rows,
                grid_cols=args.grid_cols,
            )

            all_person_results.extend(fold_results)

    excluded_for_aggregate =  ['2025_06_02_11_09_16', '2025_05_27_10_57_49', '2025_06_07_22_33_55']

    included_for_aggregate = [
        r for r in all_person_results if r["person_id"] not in excluded_for_aggregate
    ]

    if included_for_aggregate:
        accuracies = [r["metrics"]["accuracy"] for r in included_for_aggregate]
        balanced_accuracies = [r["metrics"]["balanced_accuracy"] for r in included_for_aggregate]
        f1_macros = [r["metrics"]["f1_macro"] for r in included_for_aggregate]
        precision_macros = [r["metrics"]["precision_macro"] for r in included_for_aggregate]
        recall_macros = [r["metrics"]["recall_macro"] for r in included_for_aggregate]

        aggregate_metrics = {
            "accuracy_mean": float(np.mean(accuracies)),
            "accuracy_std": float(np.std(accuracies)),
            "balanced_accuracy_mean": float(np.mean(balanced_accuracies)),
            "balanced_accuracy_std": float(np.std(balanced_accuracies)),
            "f1_macro_mean": float(np.mean(f1_macros)),
            "f1_macro_std": float(np.std(f1_macros)),
            "precision_macro_mean": float(np.mean(precision_macros)),
            "precision_macro_std": float(np.std(precision_macros)),
            "recall_macro_mean": float(np.mean(recall_macros)),
            "recall_macro_std": float(np.std(recall_macros)),
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
        "grid_rows": args.grid_rows,
        "grid_cols": args.grid_cols,
        "per_person_results": all_person_results,
        "aggregate_metrics": aggregate_metrics,
    }

    os.makedirs("experiments", exist_ok=True)
    summary_path = os.path.join("experiments", f"run_classifier_loso_summary_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json",)

    with open(summary_path, "w") as f: json.dump(summary, f, indent=2)

    print(f"Global LOSO summary saved to {summary_path}")

if __name__ == '__main__':
    main()


