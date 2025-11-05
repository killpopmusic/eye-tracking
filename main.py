import argparse
import os
import torch

from src.train import train_model
from src.evaluate import evaluate_model
from src.utils.data_loader import get_data_loaders

def main():
    parser = argparse.ArgumentParser(description="Gaze Tracking Model Training and Evaluation")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate'], help='Mode to run the script in.')
    parser.add_argument('--model_name', type=str, default='TestModel', help='Name of the model class to use.')
    parser.add_argument('--data_path', type=str, default='data/merged_dataset.json', help='Path to the training/evaluation data.')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and validation.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate for the optimizer.')
    parser.add_argument('--model_path', type=str, default='models/gaze_model.pth', help='Path to save or load the model.')

    args = parser.parse_args()

    # Dynamically import the model class
    model_class = None
    if args.model_name == 'TestModel':
        from src.models.test_model import TestModel
        model_class = TestModel
    else:
        print(f"Model {args.model_name} not found.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.mode == 'train':
        print(f"Starting training on {device}...")
        train_loader, val_loader, input_features = get_data_loaders(data_path=args.data_path, batch_size=args.batch_size)
        model = model_class(input_features).to(device)
        
        train_model(model, train_loader, val_loader, num_epochs=args.epochs, lr=args.lr, model_path=args.model_path)
        
        print("Training finished. Starting evaluation...")
        evaluate_model(model, val_loader, args.model_path)
        print("Evaluation finished.")

    elif args.mode == 'evaluate':
        if not os.path.exists(args.model_path):
            print(f"Model file not found at {args.model_path}")
            return
            
        print(f"Starting evaluation on {device}...")
        
        # Load checkpoint and initialize model
        checkpoint = torch.load(args.model_path, map_location=device)
        model = model_class(checkpoint['input_features']).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])

        _, val_loader, _ = get_data_loaders(data_path=args.data_path, batch_size=args.batch_size)
        
        evaluate_model(model, val_loader, args.model_path)
        print("Evaluation finished.")

if __name__ == '__main__':
    main()