import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import GradScaler, autocast
import matplotlib.pyplot as plt
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

def train_regressor(model, train_loader, val_loader, num_epochs, lr, model_path, patience=30):
    model.to(device)

    loss_fn = nn.SmoothL1Loss()  
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10)
    scaler = GradScaler(enabled=(device == 'cuda'))

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    train_losses = []
    val_losses = []
    val_maes = []

    best_val_loss = np.inf
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        # regression: no accuracy
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            
            with autocast("cuda"):
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
        
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        running_val_loss = 0.0

        with torch.inference_mode():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                with autocast("cuda"):
                    outputs = model(inputs)
                    loss = loss_fn(outputs, labels)
                running_val_loss += loss.item()

                batch_mae = torch.mean(torch.abs(outputs - labels)).item()
                
                    
        val_loss = running_val_loss / len(val_loader)
        val_losses.append(val_loss)
        val_maes.append(batch_mae)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val MAE: {batch_mae:.4f}")

        scheduler.step(val_loss)

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
    plt.title('Training and Validation Loss (Regressor)')
    plt.legend()
    plt.savefig('plot/loss_curve_classifier.png')
    
    plt.figure(figsize=(10, 5))
    plt.plot(val_maes, label='Validation MAE')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.title('Validation MAE (Regressor)')
    plt.legend()
    plt.savefig('plot/mae_curve_regressor.png')

    if best_model_state:
        save_dict = {
            'model_state_dict': best_model_state,
            'input_features': model.input_features,
            'output_dim': getattr(model, 'output_dim', 2)
        }
        torch.save(save_dict, model_path)
        print(f"Best model saved to {model_path}")
    else:
        save_dict = {
            'model_state_dict': model.state_dict(),
            'input_features': model.input_features,
            'output_dim': getattr(model, 'output_dim', 2)
        }
        torch.save(save_dict, model_path)
        print(f"Model saved to {model_path}")

def fine_tune_model(model, train_loader, epochs=10, lr=0.0001):
    #1st linear layer only!
    device = next(model.parameters()).device
    model.train()

    for param in model.parameters():
        param.requires_grad = False
    
    trainable_params = []
    first_linear = None

    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            first_linear = module
            break
            
    if first_linear:
        for param in first_linear.parameters():
            param.requires_grad = True
            trainable_params.append(param)
        print("Fine-tuning strategy: Optimizing FIRST Layer (Input Adaptation).")
    
    if not trainable_params:
        print("Error: No trainable parameters found.")
        return

    optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=0.001)
    criterion = torch.nn.SmoothL1Loss()

    for _ in range(epochs):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()