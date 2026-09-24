import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(
        self,input_channels=3,num_classes=10,dropout=0.15,kernel_size=3):
        super().__init__()

        self.conv1 = nn.Conv2d(
            input_channels,32,kernel_size=kernel_size,stride=1,padding="same"
        )

        self.conv2 = nn.Conv2d(32,64,kernel_size=kernel_size,stride=1,padding="same")

        # 假设输入仍然是 28×28
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
