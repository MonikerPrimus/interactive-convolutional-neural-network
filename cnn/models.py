import torch.nn as nn

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

# Define the main CNN with a single branch and merging
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