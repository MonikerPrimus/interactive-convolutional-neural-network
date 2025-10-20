import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import random
from models import MergedCNN


# Transform compose
aug_transform = transforms.Compose([
    transforms.RandomRotation(degrees=20),
    transforms.RandomResizedCrop(size=(28, 28), scale=(0.75, 1.25)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
normal_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_norm_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=normal_transform)
train_aug_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=aug_transform)
train_norm_loader = DataLoader(train_norm_dataset, batch_size=64, shuffle=True)
train_aug_loader = DataLoader(train_aug_dataset, batch_size=64, shuffle=True)

# Init  model, loss function, and optimizer
model = MergedCNN()
model.train()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

total_params = sum(p.numel() for p in model.parameters())
print("Total number of parameters in the model:", total_params)

# Training loop
# Adjust epoch
num_epochs = 10
for epoch in range(num_epochs):
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_aug_loader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Print progress every 100 batches
        if (batch_idx + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_aug_loader)}], Loss: {loss.item():.4f}")

    accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%")



# Test the trained model on a random MNIST image
model.eval()  # Set the model to eval mode
with torch.no_grad():
    random_index = random.randint(0, len(train_aug_dataset) - 1)
    random_image, random_label = train_aug_dataset[random_index]
    random_image = random_image.unsqueeze(0)  # Add batch dimension
    random_output = model(random_image)

    _, predicted_label = torch.max(random_output, 1)

    plt.imshow(random_image.squeeze().numpy(), cmap="gray")
    plt.title(f"True Label: {random_label}, Predicted Label: {predicted_label.item()}")
    plt.show()


# Save the trained model
torch.save(model.state_dict(), 'merged_cnn_model.pth')