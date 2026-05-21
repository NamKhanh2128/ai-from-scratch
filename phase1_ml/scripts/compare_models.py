import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('phase1_ml/reports/phase1_results.csv')
print("Current results:\n", df)

models = df['model'].values
metrics = ['accuracy', 'precision', 'recall', 'f1']
x = np.arange(len(models))
width = 0.2
fig, ax = plt.subplots(figsize=(10,6))
for i, metric in enumerate(metrics):
    values = df[metric].values
    ax.bar(x + i*width, values, width, label=metric.capitalize())
ax.set_xticks(x + width*1.5)
ax.set_xticklabels(models)
ax.set_ylim(0,1)
ax.set_ylabel('Score')
ax.set_title('Model Comparison on Test Set')
ax.legend()
plt.tight_layout()
plt.savefig('phase1_ml/reports/figures/metric_comparison.png')
plt.close()
print("Comparison chart saved to phase1_ml/reports/figures/metric_comparison.png")
