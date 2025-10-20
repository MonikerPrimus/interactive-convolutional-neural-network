import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Define a simple CNN block
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.4):
        super(CNNBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout2d(dropout_rate)  # Layer-wise dropout
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout2d(dropout_rate)  # Layer-wise dropout
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.pool(x)

        return x

# Define the main CNN with a single branch and merge
class MergedCNN(nn.Module):
    def __init__(self, dropout_rate=0.4):
        super(MergedCNN, self).__init__()

        # Branch
        self.branch = CNNBlock(1, 32, dropout_rate)

        # Merging
        self.merge = nn.Conv2d(32, 64, kernel_size=1)
        self.relu = nn.ReLU()

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 14 * 14, 256)
        self.dropout = nn.Dropout(dropout_rate)  # Fully connected layer dropout
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        # Branch
        out_branch = self.branch(x)

        # Merge
        out = self.merge(out_branch)
        out = self.relu(out)

        # Flatten
        out = out.view(out.size(0), -1)

        # Fully connected layers
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out
    

# Load MNIST dataset
transform = transforms.Compose([
    transforms.RandomRotation(degrees=20),
    transforms.RandomResizedCrop(size=(28, 28), scale=(0.75, 1.25)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

full_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Whole dataset for training
train_size = len(full_dataset)
validation_size = train_size // 2  # Use half of the dataset for validation
train_dataset = full_dataset
validation_dataset, _ = random_split(full_dataset, [validation_size, len(full_dataset) - validation_size])


# Create DataLoader instances for training and validation
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=64, shuffle=True)

# Optim: grid search for learning rate and weight decay
# lr_values - 0.001, 0.005, 0.01, 0.05, 0.1
# wd_values - 1e-5, 1e-4, 1e-3, 0
learning_rate_values = [0.005] 
weight_decay_values = [1e-5]  
best_learning_rate = None
best_weight_decay = None
best_accuracy = 0.0
data_set_accuracy = [
                      ]

for lr in learning_rate_values:
    lr_accuracy = []
    for weight_decay in weight_decay_values:
        model = MergedCNN()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()

        num_epochs = 10
        for epoch in range(num_epochs):
            correct = 0
            total = 0

            for batch_idx, (images, labels) in enumerate(train_loader):
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
                    print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}, lr = {lr}, weight_decay = {weight_decay}")

            accuracy = 100 * correct / total
            lr_accuracy.append(f"Epoch: {epoch+1}, Accuracy: {accuracy:.2f}")
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%")


        # Validation
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in validation_loader: 
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            print(f"Validation Accuracy with lr={lr} and weight decay={weight_decay}: {accuracy:.2f}%")

            lr_accuracy.append(f"lr={lr} and weight decay={weight_decay}: Accuracy: {accuracy:.2f}%")

            # Check if current lr and weight decay give better accuracy
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_learning_rate = lr
                best_weight_decay = weight_decay
                print(f"CURRENT best learning rate: {best_learning_rate}, Best weight decay: {best_weight_decay} with accuracy: {best_accuracy:.2f}%")
    data_set_accuracy.append(lr_accuracy)
    
print(f"Best learning rate: {best_learning_rate}, Best weight decay: {best_weight_decay} with accuracy: {best_accuracy:.2f}%")


print(data_set_accuracy)

# Train the final model with the best learning rate and weight decay
model = MergedCNN()
optimizer = optim.Adam(model.parameters(), lr=best_learning_rate, weight_decay=best_weight_decay)



# Save the trained model
torch.save(model.state_dict(), 'merged_cnn_model.pth')