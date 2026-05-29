# scripts/visualize_phase2_batch.py
import torch
import matplotlib.pyplot as plt
import numpy as np
from src.intern_ai.datasets import get_cifar10_loaders

def imshow(img, mean, std):
    # Đảo ngược normalization để hiển thị
    img = img.numpy().transpose((1, 2, 0))
    img = std * img + mean   # unnormalize
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    plt.axis('off')

def main():
    train_loader, _, _ = get_cifar10_loaders(batch_size=16)
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2023, 0.1994, 0.2010])
    
    # Lấy một batch
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    
    # Hiển thị 16 ảnh
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        imshow(images[i], mean, std)
        ax.set_title(classes[labels[i]])
    plt.tight_layout()
    plt.savefig('reports/figures/phase2_sample_batch.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    main()
