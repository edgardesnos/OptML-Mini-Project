import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import STL10
from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset, random_split, Subset
from torch.nn.functional import interpolate
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
import torch.nn.functional as F
import random
import os
import gc



def get_data_loader(batch_size=32):
    # Get STL10 dataset

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(96, padding=12),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = STL10(root='./data', split='train', download=True, transform=transform)
    testset = STL10(root='./data', split='test', download=True, transform=transform)


    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

    return trainloader, testloader



def initialize_model(out_dim=10):
    # Define the Model

    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
    model.fc = nn.Linear(2048, out_dim)  # STL-10 has 10 classes
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")

    return model



def train(model, trainloader, num_epochs, device):
    # Post-Training over 5k images from STL10

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Resize to 224×224 before feeding into ResNet
            inputs_resized = interpolate(inputs, size=(224, 224), mode='bilinear', align_corners=False)

            optimizer.zero_grad()
            outputs = model(inputs_resized)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(trainloader)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")



def test(model, testloader, device):

    # Performance over small test set
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs_resized = interpolate(inputs, size=(224, 224), mode='bilinear', align_corners=False)
            outputs = model(inputs_resized)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")



if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainloader, testloader = get_data_loader(batch_size=32)

    model = initialize_model(out_dim=10)

    train(model, trainloader, 32, device)

    test(model, testloader, device)

    # Save the model
    model_filename = "resnet50_stl10.pth"
    current_dir = os.getcwd()

    model_path = os.path.join(current_dir, model_filename)
    torch.save(model.state_dict(), model_path)

    print(f"Model saved to: {model_path}")

    # Clean up memory
    del model
    gc.collect()
    torch.cuda.empty_cache()

    






