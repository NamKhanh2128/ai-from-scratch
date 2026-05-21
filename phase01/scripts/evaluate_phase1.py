import argparse, yaml, pickle, json, os
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--scaler-path', default=None)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    with open(args.model_path, 'rb') as f:
        model = pickle.load(f)

    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')
    test_size = config.get('test_size', 0.2)
    val_size = config.get('val_size', 0.15)
    random_state = config.get('random_state', 42)
    stratify = config.get('stratify', True)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y if stratify else None
    )
    if val_size > 0:
        val_ratio = val_size / (1 - test_size)
        _, X_test, _, y_test = train_test_split(
            X_temp, y_temp, test_size=1-val_ratio, random_state=random_state,
            stratify=y_temp if stratify else None
        )
    else:
        X_test, y_test = X_temp, y_temp

    if args.scaler_path:
        with open(args.scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        X_test = scaler.transform(X_test)

    y_pred = model.predict(X_test)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }

    exp_name = config.get('name', 'experiment')
    print(f"=== Evaluation for {exp_name} ===")
    for k, v in metrics.items():
        if k != 'confusion_matrix':
            print(f"{k}: {v:.4f}")
    print("Confusion matrix:\n", np.array(metrics['confusion_matrix']))

    os.makedirs('phase1_ml/reports', exist_ok=True)
    with open(f'phase1_ml/reports/metrics_{exp_name}.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    csv_path = 'phase1_ml/reports/phase1_results.csv'
    df_new = pd.DataFrame([{
        'model': exp_name,
        **{k:v for k,v in metrics.items() if k != 'confusion_matrix'}
    }])
    if os.path.exists(csv_path):
        df_old = pd.read_csv(csv_path)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(csv_path, index=False)

    os.makedirs('phase1_ml/reports/figures', exist_ok=True)
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=['malignant','benign'], yticklabels=['malignant','benign'])
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix - {exp_name}')
    plt.tight_layout()
    fig_path = f'phase1_ml/reports/figures/confusion_matrix_{exp_name}.png'
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved confusion matrix to {fig_path}")
    print(f"Results appended to {csv_path}")

if __name__ == "__main__":
    main()
