import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

class AblationVisualizer:
    """Ablation study visualizer."""
    
    def __init__(self, output_dir):
        """Initialize visualizer."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.colors = {
            'baseline': '#1f77b4',
            'equal_weight': '#ff7f0e',
            'fixed_30day': '#2ca02c',
            'literature_rl': '#d62728',
            'mean_var': '#9467bd',
            'momentum': '#8c564b',
            'no_cov_penalty': '#e377c2',
            'no_exit_rule': '#7f7f7f',
            'no_LIM': '#bcbd22'
        }
        
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['mathtext.fontset'] = 'stix'
        
    def plot_ablation_portfolio_values(self, results_dict, title=None, save_path=None):
        """Plot portfolio value curves for ablation studies."""
        plt.figure(figsize=(12, 6))
        
        for strategy, data in results_dict.items():
            if 'date' in data and 'portfolio_value' in data:
                initial_value = data['portfolio_value'][0]
                normalized_values = [v/initial_value for v in data['portfolio_value']]
                
                plt.plot(pd.to_datetime(data['date']), normalized_values, 
                        label=strategy.replace('_', ' ').title(), 
                        color=self.colors.get(strategy, '#333333'),
                        linewidth=2)
        
        if title:
            plt.title(title, fontsize=14, fontfamily='Times New Roman', fontweight='bold')
        else:
            plt.title('Portfolio Value Comparison (Ablation Study)', 
                     fontsize=14, fontfamily='Times New Roman', fontweight='bold')
            
        plt.xlabel('Date', fontsize=12, fontfamily='Times New Roman')
        plt.ylabel('Normalized Portfolio Value', fontsize=12, fontfamily='Times New Roman')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
        plt.gcf().autofmt_xdate()
        
        plt.xticks(fontsize=10, fontfamily='Times New Roman')
        plt.yticks(fontsize=10, fontfamily='Times New Roman')
        
        if len(results_dict) > 0:
            plt.legend(loc='upper left', fontsize=10, frameon=True, fancybox=True, framealpha=0.8)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, f'ablation_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
        
    def plot_ablation_metrics(self, results_dict, metrics=None, save_path=None):
        """Plot metric comparisons for ablation studies."""
        if metrics is None:
            metrics = ['total_return', 'sharpe_ratio', 'max_drawdown']
            
        fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))
        
        if len(metrics) == 1:
            axes = [axes]
            
        for i, metric in enumerate(metrics):
            values = []
            labels = []
            
            for strategy, data in results_dict.items():
                if metric in data:
                    values.append(data[metric])
                    labels.append(strategy.replace('_', ' ').title())
            
            x = np.arange(len(labels))
            
            bars = axes[i].bar(x, values, 
                             color=[self.colors.get(label.lower().replace(' ', '_'), '#333333') for label in labels])
            
            axes[i].set_title(f'{metric.replace("_", " ").title()}', 
                            fontsize=12, fontfamily='Times New Roman')
            
            axes[i].set_xticks(x)
            axes[i].set_xticklabels(labels, rotation=45, ha='right',
                                  fontsize=10, fontfamily='Times New Roman')
            
            axes[i].tick_params(axis='y', labelsize=10)
            
            for label in axes[i].get_yticklabels():
                label.set_fontfamily('Times New Roman')
            
            for bar in bars:
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.2f}',
                           ha='center', va='bottom',
                           fontsize=10, fontfamily='Times New Roman')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, f'ablation_metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path 

