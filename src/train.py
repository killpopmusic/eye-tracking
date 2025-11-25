import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

def train_classifier(model, train_loader, val_loader, num_epochs, lr, model_path, patience=50, class_weights=None):
    model.to(device)
    
    if class_weights is not None:
        class_weights = class_weights.to(device)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    else:
        loss_fn = nn.CrossEntropyLoss()
        
    optimizer = optim.Adam(model.parameters(), lr=lr,  weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=20)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    train_losses = []
    val_losses = []
    val_accuracies = []

    best_val_loss = np.inf
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        train_acc = 100 * correct_train / total_train

        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.inference_mode():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                running_val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                    
        val_loss = running_val_loss / len(val_loader)
        val_losses.append(val_loss)
        val_acc = 100 * correct_val / total_val
        val_accuracies.append(val_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

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
    plt.title('Training and Validation Loss (Classifier)')
    plt.legend()
    plt.savefig('plot/loss_curve_classifier.png')
    
    plt.figure(figsize=(10, 5))
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy (Classifier)')
    plt.legend()
    plt.savefig('plot/accuracy_curve_classifier.png')

    if best_model_state:
        save_dict = {
            'model_state_dict': best_model_state,
            'input_features': model.input_features,
            'num_classes': getattr(model, 'num_classes', 9)
        }
        torch.save(save_dict, model_path)
        print(f"Best model saved to {model_path}")
    else:
        save_dict = {
            'model_state_dict': model.state_dict(),
            'input_features': model.input_features,
            'num_classes': getattr(model, 'num_classes', 9)
        }
        torch.save(save_dict, model_path)
        print(f"Model saved to {model_path}")
