import json
import glob
import pandas as pd

records = []
for f in glob.glob('reports/*_history.json'):
    with open(f, 'r') as fh:
        history = json.load(fh)
    # Lấy best val acc
    best_val_acc = max(history['val_acc'])
    # Lấy tên file làm experiment
    name = f.split('/')[-1].replace('_history.json', '')
    records.append({'experiment': name, 'best_val_acc': best_val_acc,
                    'final_train_loss': history['train_loss'][-1],
                    'final_val_loss': history['val_loss'][-1]})
df = pd.DataFrame(records)
df.to_csv('reports/phase2_experiment_comparison.csv', index=False)
print(df)
