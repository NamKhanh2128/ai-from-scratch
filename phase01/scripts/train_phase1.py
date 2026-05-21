import argparse, yaml, pickle, json, os
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def load_data(config):
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
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=1-val_ratio, random_state=random_state,
            stratify=y_temp if stratify else None
        )
    else:
        X_val, y_val = None, None
        X_test, y_test = X_temp, y_temp
    return X_train, X_val, X_test, y_train, y_val, y_test

def get_model(config):
    model_name = config['model']
    params = config.get('params', {})
    if model_name == 'LogisticRegression':
        return LogisticRegression(**params)
    elif model_name == 'DecisionTreeClassifier':
        return DecisionTreeClassifier(**params)
    elif model_name == 'RandomForestClassifier':
        return RandomForestClassifier(**params)
    else:
        raise ValueError(f"Unknown model: {model_name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print(f"===== Experiment: {config.get('name', 'unnamed')} =====")
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(config)
    print(f"Dataset loaded: {len(X_train)+len(X_val) if X_val is not None else 0+len(X_test)} samples")

    param_list = config.get('param_grid', [config.get('params', {})])
    best_model = None
    best_score = -1
    best_scaler = None
    best_params = None

    for params in param_list:
        cfg_copy = config.copy()
        cfg_copy['params'] = params
        model = get_model(cfg_copy)
        scaler = None
        X_tr = X_train.copy()
        X_v = X_val.copy() if X_val is not None else None
        X_te = X_test.copy()

        if config.get('use_scaler', False):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_train)
            if X_v is not None:
                X_v = scaler.transform(X_val)
            X_te = scaler.transform(X_test)

        model.fit(X_tr, y_train)
        if X_v is not None:
            y_pred = model.predict(X_v)
            f1 = f1_score(y_val, y_pred)
        else:
            y_pred = model.predict(X_te)
            f1 = f1_score(y_test, y_pred)

        print(f"  Params {params} -> Val F1: {f1:.4f}")
        if f1 > best_score:
            best_score = f1
            best_model = model
            best_params = params
            best_scaler = scaler

    print(f"Best params: {best_params}, Val F1: {best_score:.4f}")
    X_te = X_test.copy()
    if best_scaler:
        X_te = best_scaler.transform(X_test)
    y_pred_test = best_model.predict(X_te)
    test_metrics = {
        'accuracy': accuracy_score(y_test, y_pred_test),
        'precision': precision_score(y_test, y_pred_test),
        'recall': recall_score(y_test, y_pred_test),
        'f1': f1_score(y_test, y_pred_test),
        'confusion_matrix': confusion_matrix(y_test, y_pred_test).tolist()
    }
    print("Test metrics:", {k:v for k,v in test_metrics.items() if k != 'confusion_matrix'})
    print("Confusion matrix:\n", np.array(test_metrics['confusion_matrix']))

    os.makedirs('phase1_ml/outputs', exist_ok=True)
    model_path = config.get('output_model', f"phase1_ml/outputs/{config['name']}_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    if best_scaler:
        scaler_path = model_path.replace('.pkl', '_scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(best_scaler, f)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
