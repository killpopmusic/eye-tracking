import argparse
import os
import torch
import h5py
import numpy as np

from src.train import train_classifier
from src.evaluate import evaluate_classifier
from src.utils.h5_data_loader import get_h5_data_loaders
from src.models.gaze_classifier import GazeClassifier
from src.models.gaze_res_mlp import GazeResMLP


def main():
    parser = argparse.ArgumentParser(description="Gaze Classification Model Training and Evaluation")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate'], help='Mode to run the script in.')
    parser.add_argument('--model_name', type=str, default='GazeClassifier', help='Name of the model class to use.')
    parser.add_argument('--data_path', type=str, default='data/HybridGaze.h5', help='Path to the training/evaluation data.')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training and validation.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate for the optimizer.')
    parser.add_argument('--model_path', type=str, default='models/gaze_classifier.pth', help='Path to save or load the model.')
    parser.add_argument('--grid_rows', type=int, default=1, help='Number of rows in the classification grid.')
    parser.add_argument('--grid_cols', type=int, default=5, help='Number of columns in the classification grid.')

    args = parser.parse_args()

    num_classes = args.grid_rows * args.grid_cols

    model_map = {
        'GazeClassifier': GazeClassifier,
        'GazeResMLP': GazeResMLP,
    }
    model_class = model_map.get(args.model_name)
    
    if not model_class:
        print(f"Model {args.model_name} not found.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with h5py.File(args.data_path, 'r') as f:
        all_person_ids = np.unique([pid.decode('utf-8') for pid in f['data']['person_id'][:]])

    train_person_ids = all_person_ids[[0,2,3,4,5,6,7,8,9,10,11,12,13,14]]
    test_person_ids = all_person_ids[[1]]

    hyperparameters = {
        "epochs": args.epochs, 
        "batch_size": args.batch_size, 
        "learning_rate": args.lr,
        "grid_size": f"{args.grid_rows}x{args.grid_cols}"
    }

    data_loader_params = {
        'data_path': args.data_path,
        'batch_size': args.batch_size,
        'train_person_ids': train_person_ids,
        'test_person_ids': test_person_ids,
        'grid_rows': args.grid_rows,
        'grid_cols': args.grid_cols
    }
    
    train_loader, val_loader, test_loader, input_features, source_csv_test, y_test, gaze_test, person_ids_test = get_h5_data_loaders(**data_loader_params)

    model = model_class(input_features, num_classes=num_classes).to(device)

    if args.mode == 'train':
        train_classifier(model, train_loader, val_loader, num_epochs=args.epochs, lr=args.lr, model_path=args.model_path)
        print("Classifier training finished. Starting evaluation on the test set...")
        evaluate_classifier(model, test_loader, args.model_path, source_csv_test, y_test, gaze_test, person_ids_test, hyperparameters, grid_rows=args.grid_rows, grid_cols=args.grid_cols)

    elif args.mode == 'evaluate':
        if not os.path.exists(args.model_path):
            print(f"Model file not found at {args.model_path}")
            return
        
        evaluate_classifier(model, test_loader, args.model_path, source_csv_test, y_test, gaze_test, person_ids_test, hyperparameters, grid_rows=args.grid_rows, grid_cols=args.grid_cols)

if __name__ == '__main__':
    main()


