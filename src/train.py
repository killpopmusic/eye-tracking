import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

def train_model(model, train_loader, val_loader, num_epochs, lr, model_path, patience=60):
    model.to(device)
    loss_fn = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4) #weight decay to prevent overfitting
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=20)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    train_losses = []
    val_losses = []

    best_val_loss = np.inf
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        running_val_loss = 0.0

        with torch.inference_mode():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                running_val_loss += loss.item()
        val_loss = running_val_loss / len(val_loader)
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        scheduler.step(val_loss)

        # early stop
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()
            print(f"New best validation loss: {best_val_loss:.4f}. Saving model state.")
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    os.makedirs('plot', exist_ok=True)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.ylim(0, 0.3)
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('plot/loss_curve.png')

    if best_model_state:
        torch.save({
            'model_state_dict': best_model_state,
            'input_features': model.input_features,
        }, model_path)
        print(f"Best model saved to {model_path}")
    else:
        torch.save({
            'model_state_dict': model.state_dict(),
            'input_features': model.input_features,
        }, model_path)
        print(f"Model saved to {model_path}")
