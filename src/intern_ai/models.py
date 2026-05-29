# src/intern_ai/models.py
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class SmallCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Block 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        # Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        # Classifier
        self.fc1 = nn.Linear(64 * 8 * 8, 256)   # sau 2 lần pool: 32->16->8
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)   # flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class PretrainedResNet18(nn.Module):
    def __init__(self, num_classes=10, freeze_backbone=True):
        super().__init__()
        self.model = models.resnet18(pretrained=True)

        # Fix cho CIFAR (32x32)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()

        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

        if freeze_backbone:
            for name, param in self.model.named_parameters():
                if 'fc' not in name:
                    param.requires_grad = False

    def forward(self, x):
        return self.model(x)
