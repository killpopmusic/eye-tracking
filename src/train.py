import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from data_loader import get_data_loaders
from model import TestModel

device = "cuda" if torch.cuda.is_available() else "cpu"

# Get data loaders
train_loader, val_loader, input_features = get_data_loaders()

# Initialize the model
model = TestModel(input_features).to(device)

loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

torch.manual_seed(42)
torch.cuda.manual_seed(42)

num_epochs = 150
train_losses = []
val_losses = []

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
    
    # Validation loop
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

# Ensure directories exist
os.makedirs('plot', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Plotting the losses
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig('plot/loss_curve.png')

# Save the model
torch.save(model.state_dict(), 'models/gaze_model.pth')
print("Model saved to models/gaze_model.pth")
