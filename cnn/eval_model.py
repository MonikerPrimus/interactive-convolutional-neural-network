import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import datasets, transforms
from models import MergedCNN

# Load the trained model
model = MergedCNN()
state_dict = torch.load('merged_cnn_model.pth', weights_only=True)
model.load_state_dict(state_dict)
model.eval()

# Define the MNIST dataset transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load the MNIST validation dataset
mnist_validation = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Small subset for quick valid
mnist_validation, _ = random_split(mnist_validation, [30000, len(mnist_validation) - 30000])

# Create a DataLoader for the valid data
validation_loader = DataLoader(mnist_validation, batch_size=100, shuffle=True)

# Evalute the model on the DataLoader
def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Accuracy on the validation set: {accuracy * 100:.2f}%")

# Evaluate the model on a small MNIST data subset
evaluate_model(model, validation_loader)
