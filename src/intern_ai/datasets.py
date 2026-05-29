# src/intern_ai/datasets.py
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

def get_transforms(train=True):
    if train:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                 std=[0.2023, 0.1994, 0.2010])
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                 std=[0.2023, 0.1994, 0.2010])
        ])
    return transform

def get_cifar10_loaders(data_root='./data', batch_size=64, num_workers=2,
                        val_ratio=0.1, seed=42):
    # Train full set (50,000)
    full_train = datasets.CIFAR10(root=data_root, train=True, download=True,
                                  transform=get_transforms(train=True))
    # Test set (10,000)
    test_data = datasets.CIFAR10(root=data_root, train=False, download=True,
                                 transform=get_transforms(train=False))
    
    # Split train thành train/validation
    val_size = int(val_ratio * len(full_train))
    train_size = len(full_train) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_data, val_data = random_split(full_train, [train_size, val_size],
                                        generator=generator)
    
    # DataLoader
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader
