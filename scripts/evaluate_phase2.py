# scripts/evaluate_phase2.py
import torch
import sys
import json
import yaml
sys.path.append('.')
from src.intern_ai.datasets import get_cifar10_loaders
from src.intern_ai.models import SmallCNN
from src.intern_ai.metrics import get_predictions, compute_metrics, plot_confusion_matrix, plot_learning_curve

def main(checkpoint_path, config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Data (chỉ cần test loader)
    _, _, test_loader = get_cifar10_loaders(
        batch_size=cfg['data']['batch_size'],
        num_workers=cfg['data']['num_workers'],
        val_ratio=cfg['data']['val_ratio'],
        seed=cfg['training']['seed']
    )
    # Model
    model = SmallCNN(num_classes=cfg['model']['num_classes']).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    # Evaluate
    y_pred, y_true = get_predictions(model, test_loader, device)
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    acc, f1, cm = compute_metrics(y_true, y_pred)
    print(f"Test Accuracy: {acc:.4f}, Macro F1: {f1:.4f}")
    # Lưu metrics
    results = {'accuracy': acc, 'macro_f1': f1}
    with open('reports/phase2_cnn_results.json', 'w') as f:
        json.dump(results, f)
    # Confusion matrix
    plot_confusion_matrix(cm, classes, normalize=False,
                          title='CNN Baseline Confusion Matrix',
                          save_path='reports/figures/phase2_confusion_matrix_cnn.png')
    # Learning curve (dùng history đã lưu khi train)
    with open('reports/phase2_cnn_history.json', 'r') as f:
        history = json.load(f)
    plot_learning_curve(history, save_path='reports/figures/phase2_learning_curve_cnn.png')

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python scripts/evaluate_phase2.py --checkpoint <path> --config <path>")
        sys.exit(1)
    main(sys.argv[2], sys.argv[4])
