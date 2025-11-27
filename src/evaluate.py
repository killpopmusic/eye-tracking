import os
import json
from datetime import datetime
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, balanced_accuracy_score, precision_recall_fscore_support

def evaluate_classifier(model, test_loader, model_path, source_csv_test, y_test, gaze_test, person_ids_test, hyperparameters, grid_rows=3, grid_cols=3):
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
            _, predicted_classes = torch.max(outputs, 1)
            all_preds.extend(predicted_classes.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    unique_person_ids = np.unique(person_ids_test)

    for person_id in unique_person_ids:
        person_mask = person_ids_test == person_id
        person_labels = all_labels[person_mask]
        person_preds = all_preds[person_mask]
        
        print(f"\n--- Evaluating Classifier for Person: {person_id} ---")

        accuracy = accuracy_score(person_labels, person_preds)
        balanced_acc = balanced_accuracy_score(person_labels, person_preds)

        # precision, recall, f1
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            person_labels,
            person_preds,
            labels=np.arange(grid_rows * grid_cols),
            average="macro",
            zero_division=0,
        )

        cm = confusion_matrix(person_labels, person_preds, labels=np.arange(grid_rows * grid_cols))

        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Balanced Accuracy: {balanced_acc:.4f}")
        print(f"Macro F1: {f1_macro:.4f}")

        run_summary = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "model_name": type(model).__name__,
            "person_id": person_id,
            "hyperparameters": hyperparameters,
            "metrics": {
                "accuracy": float(accuracy),
                "balanced_accuracy": float(balanced_acc),
                "precision_macro": float(precision_macro),
                "recall_macro": float(recall_macro),
                "f1_macro": float(f1_macro),
            },
            "confusion_matrix": cm.tolist(),
        }
        
        plot_dir = f'plot/{person_id}'
        os.makedirs(plot_dir, exist_ok=True)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(grid_rows*grid_cols), yticklabels=np.arange(grid_rows*grid_cols))
        plt.title(f'Confusion Matrix - Person {person_id}')
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        cm_path = os.path.join(plot_dir, 'confusion_matrix.png')
        plt.savefig(cm_path)
        print(f"Confusion matrix for person {person_id} saved to {cm_path}")
        plt.close()

        run_file = os.path.join('experiments', f"run_classifier_{person_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json")
        try:
            with open(run_file, 'w') as f:
                json.dump(run_summary, f, indent=2)
            print(f"Test run summary for {person_id} saved to {run_file}")
        except Exception as e:
            print(f"Failed to save test run summary JSON for {person_id}: {e}")
