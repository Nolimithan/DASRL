import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

class AblationVisualizer:
    """消融实验可视化器类"""
    
    def __init__(self, output_dir):
        """初始化可视化器
        
        Args:
            output_dir (str): 输出目录路径
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置颜色方案
        self.colors = {
            'baseline': '#1f77b4',      # 蓝色
            'equal_weight': '#ff7f0e',  # 橙色
            'fixed_30day': '#2ca02c',   # 绿色
            'literature_rl': '#d62728',  # 红色
            'mean_var': '#9467bd',      # 紫色
            'momentum': '#8c564b',      # 棕色
            'no_cov_penalty': '#e377c2', # 粉色
            'no_exit_rule': '#7f7f7f',  # 灰色
            'no_LIM': '#bcbd22'         # 黄绿色
        }
        
        # 设置全局字体
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['mathtext.fontset'] = 'stix'  # 数学公式字体
        
    def plot_ablation_portfolio_values(self, results_dict, title=None, save_path=None):
        """绘制不同消融实验的投资组合价值曲线"""
        plt.figure(figsize=(12, 6))
        
        # 绘制每个策略的投资组合价值曲线
        for strategy, data in results_dict.items():
            if 'date' in data and 'portfolio_value' in data:
                # 计算归一化的投资组合价值
                initial_value = data['portfolio_value'][0]
                normalized_values = [v/initial_value for v in data['portfolio_value']]
                
                # 绘制曲线
                plt.plot(pd.to_datetime(data['date']), normalized_values, 
                        label=strategy.replace('_', ' ').title(), 
                        color=self.colors.get(strategy, '#333333'),
                        linewidth=2)
        
        # 设置图表属性
        if title:
            plt.title(title, fontsize=14, fontfamily='Times New Roman', fontweight='bold')
        else:
            plt.title('Portfolio Value Comparison (Ablation Study)', 
                     fontsize=14, fontfamily='Times New Roman', fontweight='bold')
            
        plt.xlabel('Date', fontsize=12, fontfamily='Times New Roman')
        plt.ylabel('Normalized Portfolio Value', fontsize=12, fontfamily='Times New Roman')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 设置x轴日期格式
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
        plt.gcf().autofmt_xdate()  # 自动旋转日期标签
        
        # 设置刻度标签字体
        plt.xticks(fontsize=10, fontfamily='Times New Roman')
        plt.yticks(fontsize=10, fontfamily='Times New Roman')
        
        # 添加图例（只有在有数据时才添加）
        if len(results_dict) > 0:
            plt.legend(loc='upper left', fontsize=10, frameon=True, fancybox=True, framealpha=0.8)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        if save_path is None:
            save_path = os.path.join(self.output_dir, f'ablation_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
        
    def plot_ablation_metrics(self, results_dict, metrics=None, save_path=None):
        """绘制不同消融实验的性能指标对比图"""
        if metrics is None:
            metrics = ['total_return', 'sharpe_ratio', 'max_drawdown']
            
        # 创建子图
        fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))
        
        # 确保axes是列表
        if len(metrics) == 1:
            axes = [axes]
            
        # 为每个指标创建柱状图
        for i, metric in enumerate(metrics):
            values = []
            labels = []
            
            for strategy, data in results_dict.items():
                if metric in data:
                    values.append(data[metric])
                    labels.append(strategy.replace('_', ' ').title())
            
            # 设置x轴刻度位置
            x = np.arange(len(labels))
            
            # 创建柱状图
            bars = axes[i].bar(x, values, 
                             color=[self.colors.get(label.lower().replace(' ', '_'), '#333333') for label in labels])
            
            # 设置标题和标签
            axes[i].set_title(f'{metric.replace("_", " ").title()}', 
                            fontsize=12, fontfamily='Times New Roman')
            
            # 设置x轴刻度和标签
            axes[i].set_xticks(x)
            axes[i].set_xticklabels(labels, rotation=45, ha='right',
                                  fontsize=10, fontfamily='Times New Roman')
            
            # 设置y轴刻度标签
            axes[i].tick_params(axis='y', labelsize=10)
            
            # 设置y轴刻度标签字体
            for label in axes[i].get_yticklabels():
                label.set_fontfamily('Times New Roman')
            
            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.2f}',
                           ha='center', va='bottom',
                           fontsize=10, fontfamily='Times New Roman')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        if save_path is None:
            save_path = os.path.join(self.output_dir, f'ablation_metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path 

