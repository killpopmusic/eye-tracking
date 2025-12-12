import os
import json
from datetime import datetime
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_regressor(
    model,
    test_loader,
    model_path,
    person_ids_test,
    source_csv_test,
    hyperparameters,
    target_space: str = "normalized",
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    model.to(device)
    model.eval()

    preds_list = []
    labels_list = []

    with torch.inference_mode():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds_list.append(outputs.cpu().numpy())
            labels_list.append(labels.cpu().numpy())

    preds = np.concatenate(preds_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    # Denormalize if necessary to pixel space for metrics
    SCREEN_W, SCREEN_H = 2560.0, 1440.0
    if target_space == "normalized":
        preds_px = np.stack([preds[:, 0] * SCREEN_W, preds[:, 1] * SCREEN_H], axis=-1)
        labels_px = np.stack([labels[:, 0] * SCREEN_W, labels[:, 1] * SCREEN_H], axis=-1)
    else:
        preds_px = preds
        labels_px = labels

    #  Angular accuracy

    diag_cm = 68.58 #for 27inch
    aspect_w, aspect_h = 16.0, 9.0
    diag_factor = (aspect_w ** 2 + aspect_h ** 2) ** 0.5
    screen_w_cm = diag_cm * aspect_w / diag_factor
    screen_h_cm = diag_cm * aspect_h / diag_factor

    #pixel size in cm
    cm_per_px_x = screen_w_cm / SCREEN_W
    cm_per_px_y = screen_h_cm / SCREEN_H

    #gt and preds distance from screen center 
    x_true_cm = (labels_px[:, 0] - SCREEN_W / 2.0) * cm_per_px_x
    y_true_cm = (labels_px[:, 1] - SCREEN_H / 2.0) * cm_per_px_y
    x_pred_cm = (preds_px[:, 0] - SCREEN_W / 2.0) * cm_per_px_x
    y_pred_cm = (preds_px[:, 1] - SCREEN_H / 2.0) * cm_per_px_y

    # 3D gaze vectors 
    D_cm = 60.0 #distance from the screen
    v_true = np.stack([x_true_cm, y_true_cm, np.full_like(x_true_cm, D_cm)], axis=-1)
    v_pred = np.stack([x_pred_cm, y_pred_cm, np.full_like(x_pred_cm, D_cm)], axis=-1)

    #dot_prod = norm_true * norm_pred * cos(theta) 
    
    dot_prod = np.sum(v_true * v_pred, axis=-1)
    norm_true = np.linalg.norm(v_true, axis=-1)
    norm_pred = np.linalg.norm(v_pred, axis=-1)
    cos_theta = dot_prod / (norm_true * norm_pred + 1e-8)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angular_errors_deg = np.degrees(np.arccos(cos_theta))

    results_per_person = []
    unique_persons = np.unique(person_ids_test)

    for pid in unique_persons:
        person_mask = person_ids_test == pid

        mae = mean_absolute_error(labels_px[person_mask], preds_px[person_mask])
        rmse = np.sqrt(mean_squared_error(labels_px[person_mask], preds_px[person_mask]))
        r2 = r2_score(labels_px[person_mask], preds_px[person_mask]) if labels_px[person_mask].shape[0] > 1 else float('nan')

        person_angles = angular_errors_deg[person_mask]
        within_3deg_pct = float(np.mean(person_angles <= 3.0) * 100.0) if person_angles.size > 0 else 0.0

        print(f"Person {pid} — MAE(px): {mae:.2f}, RMSE(px): {rmse:.2f}, R2: {r2:.3f}")

        # Arrays per person
        person_labels = labels_px[person_mask]
        person_preds = preds_px[person_mask]
        person_src = np.array(source_csv_test)[person_mask]

        plot_dir = os.path.join('plot', str(pid))
        os.makedirs(plot_dir, exist_ok=True)

        # 3x3 calibration plot
        mask_3x3 = person_src == 'data_3x3.csv'
        if np.any(mask_3x3):
            plt.figure(figsize=(12, 8))
            plt.scatter(person_labels[mask_3x3, 0], person_labels[mask_3x3, 1],
                        label='Dane z Tobii 5', alpha=0.6, s=100)
            plt.scatter(person_preds[mask_3x3, 0], person_preds[mask_3x3, 1],
                        label='Predykcje modelu', alpha=0.6, s=100, c='r', marker='x')
            for i in np.where(mask_3x3)[0]:
                plt.plot([person_labels[i, 0], person_preds[i, 0]],
                         [person_labels[i, 1], person_preds[i, 1]],
                         'k-', alpha=0.2)
            plt.title(f'Predykcje modelu na tle danych referencyjnych dla użytkownika: {pid}')
            plt.xlabel('X')
            plt.ylabel('Y ')
            plt.legend()
            plt.grid(True)
            plt.xlim(0, SCREEN_W)
            plt.ylim(0, SCREEN_H)
            plt.gca().invert_yaxis()
            out_path = os.path.join(plot_dir, 'test_prediction_3x3.png')
            plt.savefig(out_path)
            print(f"Test prediction visualization for 3x3 for person {pid} saved to {out_path}")
            plt.close()

        # 5x5 calibration plot
        mask_5x5 = person_src == 'data_5x5.csv'
        if np.any(mask_5x5):
            plt.figure(figsize=(12, 8))
            plt.scatter(person_labels[mask_5x5, 0], person_labels[mask_5x5, 1],
                        label='Dane z Tobii 5', alpha=0.6, s=100)
            plt.scatter(person_preds[mask_5x5, 0], person_preds[mask_5x5, 1],
                        label='Predykcje modelu', alpha=0.6, s=100, c='r', marker='x')
            for i in np.where(mask_5x5)[0]:
                plt.plot([person_labels[i, 0], person_preds[i, 0]],
                         [person_labels[i, 1], person_preds[i, 1]],
                         'k-', alpha=0.2)
            plt.title(f'Predykcje modelu na tle danych referencyjnych dla użytkownika: {pid}')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.legend()
            plt.grid(True)
            plt.xlim(0, SCREEN_W)
            plt.ylim(0, SCREEN_H)
            plt.gca().invert_yaxis()
            out_path = os.path.join(plot_dir, 'test_prediction_5x5.png')
            plt.savefig(out_path)
            print(f"Test prediction visualization for 5x5 for person {pid} saved to {out_path}")
            plt.close()

        # Smooth plot
        mask_smooth = person_src == 'data_smooth.csv'
        if np.any(mask_smooth):
            plt.figure(figsize=(12, 8))
            plt.plot(person_labels[mask_smooth, 0], person_labels[mask_smooth, 1],
                     'b-o', label='Dane z Tobii 5', markersize=5)
            plt.plot(person_preds[mask_smooth, 0], person_preds[mask_smooth, 1],
                     'r-x', label='Predykcje modelu', markersize=5)
            plt.title(f'Predykcje modelu na tle danych referencyjnych dla użytkownika: {pid}')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.legend()
            plt.grid(True)
            plt.xlim(0, SCREEN_W)
            plt.ylim(0, SCREEN_H)
            plt.gca().invert_yaxis()
            out_path = os.path.join(plot_dir, 'test_prediction_smooth.png')
            plt.savefig(out_path)
            print(f"Test prediction visualization for smooth pursuit for person {pid} saved to {out_path}")
            plt.close()

        results_per_person.append({
            "person_id": pid,
            "metrics": {
                "mae_px": float(mae),
                "rmse_px": float(rmse),
                "r2": float(r2) if not np.isnan(r2) else None,
                "within_3deg_pct": within_3deg_pct,
            },
            "hyperparameters": hyperparameters,
        })

    return results_per_person
