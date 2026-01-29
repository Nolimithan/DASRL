import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import os

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv('D:/RL experiment/LIMPPO_CNN/0419主实验结果/A股半导体做多/test_PS3TOP_20250419_141931/test_history_weights.csv')

weight_cols = [col for col in df.columns if col.startswith('weight_')]
plot_df = df[['date'] + weight_cols].copy()

plot_df['date'] = pd.to_datetime(plot_df['date'])
plot_df = plot_df.set_index('date')

save_dir = 'D:/paper figure/LIMPPO'
os.makedirs(save_dir, exist_ok=True)

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

fig, ax = plt.subplots(figsize=(7, 4))
ax.stackplot(plot_df.index, [plot_df[col] for col in weight_cols], 
             colors=colors[:len(weight_cols)], alpha=0.85)

ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
fig.autofmt_xdate(rotation=0)

for spine in ['top', 'right', 'bottom', 'left']:
    ax.spines[spine].set_visible(True)
    ax.spines[spine].set_color('black')
    ax.spines[spine].set_linewidth(1.0)

ax.tick_params(direction='out', length=4, width=1, colors='black')

ax.set_ylim(0, 1.0)
ax.set_ylabel('Weight', fontsize=12)

plt.tight_layout(pad=0.4)
weight_path = os.path.join(save_dir, 'portfolio_weights.png')
plt.savefig(weight_path, dpi=600, bbox_inches='tight')
plt.close()

fig_legend = plt.figure(figsize=(7, 0.8))
ax = fig_legend.add_subplot(111)

legend_labels = [col.replace('weight_', '') for col in weight_cols]
legend_handles = []

simplified_labels = []
for label in legend_labels:
    simplified_labels.append(label)

x_offset = 0.05
width = 0.9 / len(simplified_labels)
height = 0.4
y_pos = 0.3

ax.text(0.5, 0.9, 'Asset Legend', ha='center', fontsize=12, family='Times New Roman')

for i, label in enumerate(simplified_labels):
    x = x_offset + i * width
    rect = plt.Rectangle((x, y_pos), width*0.8, height, 
                         facecolor=colors[i], alpha=0.85, 
                         edgecolor='black', linewidth=0.5)
    ax.add_patch(rect)
    ax.text(x + width*0.4, y_pos - 0.15, label, ha='center', va='center', 
            fontsize=10, family='Times New Roman')

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

legend_path = os.path.join(save_dir, 'asset_legend.png')
fig_legend.savefig(legend_path, dpi=600, bbox_inches='tight')
plt.close()

print(f'Weights plot saved to: {weight_path}')
print(f'Legend saved to: {legend_path}') 

