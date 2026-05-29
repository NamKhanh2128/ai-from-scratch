# scripts/train_phase2.py
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import sys
sys.path.append('.')
from src.intern_ai.datasets import get_cifar10_loaders
from src.intern_ai.models import SmallCNN, PretrainedResNet18
from src.intern_ai.training import train_model

def main(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    train_loader, val_loader, _ = get_cifar10_loaders(
        batch_size=cfg['data']['batch_size'],
        num_workers=cfg['data']['num_workers'],
        val_ratio=cfg['data']['val_ratio'],
        seed=cfg['training']['seed']
    )
    # Model
    if cfg['model']['name'] == 'SmallCNN':
         model = SmallCNN(num_classes=cfg['model']['num_classes']).to(device)
    
    elif cfg['model']['name'] == 'PretrainedResNet18':
        model = PretrainedResNet18(
            num_classes=cfg['model']['num_classes'],
            freeze_backbone=cfg['model'].get('freeze_backbone', True)
        ).to(device)

    else:
        raise ValueError(f"Unknown model: {cfg['model']['name']}")
    # Loss, optimizer
    criterion = nn.CrossEntropyLoss()
    if cfg['training']['optimizer'].lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=cfg['training']['lr'],
                                weight_decay=cfg['training']['weight_decay'])
    else:
        optimizer = optim.SGD(model.parameters(), lr=cfg['training']['lr'],
                              momentum=0.9, weight_decay=cfg['training']['weight_decay'])
    scheduler = None
    # Train
    history = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        device, cfg['training']['epochs'],
        save_path='outputs/phase2/cnn_best.pt'
    )
    # Lưu history
    import json
    with open('reports/phase2_cnn_history.json', 'w') as f:
        json.dump(history, f)
    print("Training done.")

if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != '--config':
        print("Usage: python scripts/train_phase2.py --config configs/phase2_cnn.yaml")
        sys.exit(1)
    main(sys.argv[2])
