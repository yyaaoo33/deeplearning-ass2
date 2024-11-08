import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import transforms
import numpy as np
from models.resnet18 import resnet18
from models.efficientnet import efficientnet_b0
from models.mobilenet import mobilenet
from models.vgg16 import vgg16
from utils import load_cifar10

# Load CIFAR-10 data
data_dir = 'data/cifar-10-batches-py'
train_data, train_labels, val_data, val_labels, test_data, test_labels = load_cifar10(data_dir)

# Data transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Create DataLoaders
train_dataset = TensorDataset(torch.tensor(train_data).permute(0, 3, 1, 2).float(), torch.tensor(train_labels))
val_dataset = TensorDataset(torch.tensor(val_data).permute(0, 3, 1, 2).float(), torch.tensor(val_labels))
test_dataset = TensorDataset(torch.tensor(test_data).permute(0, 3, 1, 2).float(), torch.tensor(test_labels))

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Choose model
model_name = 'resnet18'  # Change to 'resnet18', 'alexnet', 'mobilenet', or 'vgg16' as needed
use_residual = False  # Only applicable for ResNet-18
is_pretrained = True  # Set to True if you want to use pretrained weights

if model_name == 'resnet18':
    model = resnet18(num_classes=10, use_residual=use_residual, is_pretrained=is_pretrained)
elif model_name == 'efficientnet_b0':
    model = efficientnet_b0(num_classes=10, is_pretrained=is_pretrained)
elif model_name == 'mobilenet':
    model = mobilenet(num_classes=10, is_pretrained=is_pretrained)
elif model_name == 'vgg16':
    model = vgg16(num_classes=10, is_pretrained=is_pretrained)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Choose optimizer and learning rate
optimizer_name = 'adam'  # Change to 'sgd' or other optimizers as needed
learning_rate = 0.001

if optimizer_name == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
elif optimizer_name == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

# Choose loss function
criterion = nn.CrossEntropyLoss()

# Log file name
log_file = f"logs/{model_name}_{use_residual}_{optimizer_name}_{learning_rate}_{is_pretrained}.log"

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20):
    with open(log_file, 'w') as f:
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
            
            epoch_loss = running_loss / len(train_loader.dataset)
            
            # Validation loss and accuracy
            model.eval()
            val_loss = 0.0
            correct = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)
                    _, preds = torch.max(outputs, 1)
                    correct += torch.sum(preds == labels.data)
            
            val_loss = val_loss / len(val_loader.dataset)
            val_acc = correct.double() / len(val_loader.dataset)
            
            # Log the results
            f.write(f"{epoch_loss:.4f} {val_loss:.4f} {val_acc:.4f}\n")
            print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')

# Test evaluation function
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data)
    
    test_acc = correct.double() / len(test_loader.dataset)
    print(f'Test Accuracy: {test_acc:.4f}')
    
    # Log the test accuracy
    with open(log_file, 'a') as f:
        f.write(f"{test_acc:.4f}\n")

# Train the model
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20)

# Evaluate the model on the test set
evaluate_model(model, test_loader)
