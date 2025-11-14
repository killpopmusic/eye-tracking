import argparse
import os
import torch
import h5py
import numpy as np

from src.train import train_model
from src.evaluate import evaluate_model
from src.utils.h5_data_loader import get_h5_data_loaders

def main():
    parser = argparse.ArgumentParser(description="Gaze Tracking Model Training and Evaluation")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate'], help='Mode to run the script in.')
    parser.add_argument('--model_name', type=str, default='TestModel2', help='Name of the model class to use.')
    parser.add_argument('--data_path', type=str, default='data/HybridGaze.h5', help='Path to the training/evaluation data.')
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training and validation.')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate for the optimizer.')
    parser.add_argument('--model_path', type=str, default='models/gaze_model.pth', help='Path to save or load the model.')

    args = parser.parse_args()

    # Dynamically import the model class
    model_class = None
    if args.model_name == 'TestModel':
        from src.models.test_model import TestModel
        model_class = TestModel
    elif args.model_name == 'TestModel2':
        from src.models.test_model_v2 import TestModelV2
        model_class = TestModelV2
    else:
        print(f"Model {args.model_name} not found.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Get all unique person IDs
    with h5py.File(args.data_path, 'r') as f:
        all_person_ids = np.unique([pid.decode('utf-8') for pid in f['data']['person_id'][:]])

    # Split person IDs for training/validation and testing
    train_person_ids = all_person_ids[[0,2,4,5,6,7,8,9,10,11,12,13,14]]  # without 0, 2, 5, 6, 7, 12, 11
    test_person_ids = all_person_ids[[1,3]]
    print(f"Training/Validation on {len(train_person_ids)} persons: {train_person_ids}")
    print(f"Testing on {len(test_person_ids)} persons: {test_person_ids}")

    hyperparameters = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "target": "gaze"
    }

    if args.mode == 'train':
        train_loader, val_loader, test_loader, input_features, source_csv_test, y_test, gaze_test, person_ids_test = get_h5_data_loaders(
            data_path=args.data_path, 
            batch_size=args.batch_size, 
            train_person_ids=train_person_ids,
            test_person_ids=test_person_ids
        )
        model = model_class(input_features).to(device)
        
        train_model(model, train_loader, val_loader, num_epochs=args.epochs, lr=args.lr, model_path=args.model_path)
        
        print("Training finished. Starting evaluation on the test set...")
        evaluate_model(model, test_loader, args.model_path, source_csv_test, y_test, gaze_test, person_ids_test, hyperparameters)

    elif args.mode == 'evaluate':
        if not os.path.exists(args.model_path):
            print(f"Model file not found at {args.model_path}")
            return
        _, _, test_loader, input_features, source_csv_test, y_test, gaze_test, person_ids_test = get_h5_data_loaders(
            data_path=args.data_path, 
            batch_size=args.batch_size,
            train_person_ids=train_person_ids,
            test_person_ids=test_person_ids
        )
        
        model = model_class(input_features).to(device)
        
        evaluate_model(model, test_loader, args.model_path, source_csv_test, y_test, gaze_test, person_ids_test, hyperparameters)

if __name__ == '__main__':
    main()


