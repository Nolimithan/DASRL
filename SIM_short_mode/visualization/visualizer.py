"""
Visualization module for strategy results.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from datetime import datetime
import matplotlib as mpl

class Visualizer:
    """Visualization helper for plotting results."""
    
    def __init__(self, output_dir='visualization'):
        """Initialize visualizer."""
        self.output_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        plt.style.use('default')
        
        plt.rcParams.update({
            'font.family': 'Times New Roman',
            'font.size': 12,
            'mathtext.fontset': 'stix',
            'axes.unicode_minus': False,
            
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'figure.titlesize': 16,
            
            'lines.linewidth': 2,
            'grid.alpha': 0.3,
            'grid.linestyle': '--',
            'axes.grid': True,
            'axes.axisbelow': True,
            
            'figure.figsize': [12, 8],
            'figure.dpi': 300,
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            
            'axes.spines.top': False,
            'axes.spines.right': False,
            
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1,
            'figure.autolayout': True,
            
            'xtick.direction': 'out',
            'ytick.direction': 'out',
            'xtick.major.width': 1.0,
            'ytick.major.width': 1.0,
            
            'legend.frameon': True,
            'legend.framealpha': 0.8,
            'legend.edgecolor': '0.8',
            'legend.fancybox': True
        })
        
        self.chinese_font = ['SimHei', 'Microsoft YaHei', 'Arial']
        
        self.colors = {
            'main': '#2878B5',
            'secondary': '#F8AC00',
            'tertiary': '#9AC9DB',
            'quaternary': '#C82423',
            'background': '#ffffff',
            'grid': '#CCCCCC',
            'text': '#333333'
        }
        
        self.fill_alpha = 0.15
        
        mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=[
            '#2878B5', '#F8AC00', '#9AC9DB', '#C82423',
            '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
            '#bcbd22', '#17becf'
        ])
        
        mpl.rcParams['text.color'] = self.colors['text']
        mpl.rcParams['axes.labelcolor'] = self.colors['text']
        mpl.rcParams['xtick.color'] = self.colors['text']
        mpl.rcParams['ytick.color'] = self.colors['text']
    
    def plot_results(self, portfolio, benchmark=None, save_path=None):
        """Plot overall strategy results."""
        history_df = portfolio.get_history_dataframe()
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 18), gridspec_kw={'height_ratios': [3, 1, 1]})
        
        self._plot_nav_curve(axes[0], history_df, benchmark)
        
        self._plot_return_curve(axes[1], history_df)
        
        self._plot_drawdown_curve(axes[2], history_df)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, f'portfolio_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.plot_holdings_heatmap(portfolio, save_path=None)
        
        self.plot_performance_radar(portfolio, save_path=None)
        
        self._print_performance_metrics(portfolio)
    
    def _plot_nav_curve(self, ax, history_df, benchmark=None):
        """Plot net asset value curve."""
        ax.set_facecolor(self.colors['background'])
        
        portfolio_nav = history_df['portfolio_value'] / history_df['portfolio_value'].iloc[0]
        
        ax.plot(history_df['date'], portfolio_nav, 
                label='lim*Enhanced Strategy', 
                color=self.colors['main'], 
                linewidth=3,
                marker='o',
                markersize=0,
                markevery=20)
        
        if benchmark is not None:
            benchmark = benchmark[benchmark['date'].isin(history_df['date'])]
            
            if not benchmark.empty:
                benchmark_nav = benchmark['close'] / benchmark['close'].iloc[0]
                
                ax.plot(benchmark['date'], benchmark_nav, 
                       label='Semiconductor ETF Benchmark', 
                       color=self.colors['secondary'], 
                       linewidth=2.5, 
                       linestyle='--',
                       marker='s',
                       markersize=0,
                       markevery=20)
        
        ax.set_title('Net Value Comparison', fontsize=16, fontweight='bold', pad=15)
        ax.set_xlabel('Date', fontsize=14, labelpad=10)
        ax.set_ylabel('Net Value (Initial=1)', fontsize=14, labelpad=10)
        
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.2f}'))
        
        ax.grid(True, linestyle='--', alpha=0.3, color=self.colors['grid'])
        
        ax.legend(loc='upper left', fontsize=12, frameon=True, framealpha=0.9)
        
        if benchmark is not None and not benchmark.empty:
            ax.fill_between(history_df['date'], 
                           portfolio_nav, 
                           benchmark_nav, 
                           where=(portfolio_nav > benchmark_nav),
                           color=self.colors['main'], 
                           alpha=self.fill_alpha,
                           interpolate=True)
        
        ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mpl.dates.MonthLocator(interval=3))
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        ax.axhline(y=1.0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    
    def _plot_return_curve(self, ax, history_df):
        """Plot cumulative return curve."""
        ax.set_facecolor(self.colors['background'])
        
        cum_return = (history_df['portfolio_value'] / history_df['portfolio_value'].iloc[0]) - 1
        
        ax.plot(history_df['date'], cum_return * 100, 
               label='Cumulative Return', 
               color=self.colors['main'], 
               linewidth=3)
        
        ax.set_title('Cumulative Return', fontsize=16, fontweight='bold', pad=15)
        ax.set_xlabel('Date', fontsize=14, labelpad=10)
        ax.set_ylabel('Return (%)', fontsize=14, labelpad=10)
        
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0f}%'))
        
        ax.grid(True, linestyle='--', alpha=0.3, color=self.colors['grid'])
        
        ax.legend(loc='upper left', fontsize=12, frameon=True, framealpha=0.9)
        
        ax.fill_between(history_df['date'], 0, cum_return * 100, 
                      color=self.colors['main'], alpha=self.fill_alpha)
        
        ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mpl.dates.MonthLocator(interval=3))
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    def _plot_drawdown_curve(self, ax, history_df):
        """Plot drawdown curve."""
        ax.set_facecolor(self.colors['background'])
        
        rolling_max = history_df['portfolio_value'].cummax()
        
        drawdown = (rolling_max - history_df['portfolio_value']) / rolling_max
        
        ax.fill_between(history_df['date'], 0, drawdown * 100, 
                      alpha=0.3, 
                      color=self.colors['quaternary'],
                      label='Drawdown')
        
        ax.plot(history_df['date'], drawdown * 100, 
               color=self.colors['main'], 
               linewidth=1)
        
        ax.set_title('Drawdown Curve', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Drawdown (%)', fontsize=12)
        
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0f}%'))
        
        ax.invert_yaxis()
        
        ax.grid(True, linestyle='--', alpha=0.7, color=self.colors['grid'])
        
        ax.legend(loc='lower left', fontsize=10)
    
    def plot_holdings_heatmap(self, portfolio, save_path=None):
        """Plot holdings heatmap."""
        holdings_history = portfolio.history['holdings']
        dates = portfolio.history['date']
        
        all_symbols = set()
        for holdings in holdings_history:
            all_symbols.update(holdings)
        all_symbols = sorted(list(all_symbols))
        
        weights_matrix = np.zeros((len(dates), len(all_symbols)))
        
        for i, (date, holdings) in enumerate(zip(dates, holdings_history)):
            weights = portfolio.history['transactions'][i]
            for j, symbol in enumerate(all_symbols):
                if symbol in weights:
                    weights_matrix[i, j] = weights[symbol]
        
        weights_df = pd.DataFrame(weights_matrix, index=dates, columns=all_symbols)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(weights_df, cmap='YlGnBu', linewidths=0.5, 
                   cbar_kws={'label': 'Portfolio Weight'})
        
        plt.title('Holdings Heatmap', fontsize=14, fontweight='bold')
        plt.xlabel('Stocks', fontsize=12)
        plt.ylabel('Date', fontsize=12)
        
        plt.xticks(rotation=45, ha='right')
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, f'holdings_heatmap_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_performance_radar(self, portfolio, save_path=None):
        """Plot performance metrics radar chart."""
        metrics = portfolio.get_performance_summary()
        
        radar_metrics = {}
        
        def safe_convert_percent(value):
            if isinstance(value, str) and '%' in value:
                return float(value.strip('%')) / 100
            else:
                return float(value)
                
        radar_metrics = {
            'Annual Return': safe_convert_percent(metrics['annual_return']),
            'Sharpe Ratio': float(metrics['sharpe_ratio']),
            'Sortino Ratio': float(metrics['sortino_ratio']),
            'Win Rate': safe_convert_percent(metrics['win_rate']),
            'Max Drawdown': -safe_convert_percent(metrics['max_drawdown']),
            'Volatility': -safe_convert_percent(metrics['volatility'])
        }
        
        min_val = min(radar_metrics.values())
        max_val = max(radar_metrics.values())
        
        for key in radar_metrics:
            radar_metrics[key] = (radar_metrics[key] - min_val) / (max_val - min_val)
        
        categories = list(radar_metrics.keys())
        values = list(radar_metrics.values())
        
        values.append(values[0])
        categories.append(categories[0])
        
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=True)
        
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
        
        ax.fill(angles, values, color=self.colors['main'], alpha=0.25)
        ax.plot(angles, values, color=self.colors['main'], linewidth=2)
        
        ax.set_thetagrids(angles * 180 / np.pi, categories, fontsize=12)
        
        ax.set_title('Performance Metrics', fontsize=14, fontweight='bold')
        
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
        ax.set_ylim(0, 1)
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, f'performance_radar_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _print_performance_metrics(self, portfolio):
        """Render a performance metrics summary table."""
        metrics = portfolio.get_performance_summary()
        
        print("Metric keys:", list(metrics.keys()))
        
        def safe_get_metric(key, alternative_keys=None):
            if key in metrics:
                return metrics[key]
            elif alternative_keys:
                for alt_key in alternative_keys:
                    if alt_key in metrics:
                        return metrics[alt_key]
            return "N/A"
        
        def format_metric(key, value):
            if isinstance(value, (int, float, np.number)):
                if key in ['total_return', 'annual_return', 'max_drawdown', 'volatility', 'win_rate', 'turnover', 'turnover_rate', 'cost_ratio', 'cost_percentage']:
                    return f"{float(value):.2f}%"
                else:
                    return f"{float(value):.4f}"
            return str(value)
        
        plt.figure(figsize=(12, 8))
        
        ax = plt.subplot(111, frame_on=False)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        
        metrics_data = [
            ['Metric', 'Value'],
            ['Total Return', format_metric('total_return', safe_get_metric('total_return'))],
            ['Annual Return', format_metric('annual_return', safe_get_metric('annual_return'))],
            ['Sharpe Ratio', format_metric('sharpe_ratio', safe_get_metric('sharpe_ratio'))],
            ['Sortino Ratio', format_metric('sortino_ratio', safe_get_metric('sortino_ratio'))],
            ['Max Drawdown', format_metric('max_drawdown', safe_get_metric('max_drawdown'))],
            ['Volatility', format_metric('volatility', safe_get_metric('volatility'))],
            ['Win Rate', format_metric('win_rate', safe_get_metric('win_rate'))],
            ['Turnover Rate', format_metric('turnover', safe_get_metric('turnover', ['turnover_rate']))],
            ['Stop Loss Count', format_metric('stop_loss_count', safe_get_metric('stop_loss_count'))],
            ['Cost Ratio', format_metric('cost_ratio', safe_get_metric('cost_ratio', ['cost_percentage']))]
        ]
        
        table = ax.table(cellText=metrics_data,
                        colWidths=[0.3, 0.2],
                        cellLoc='center',
                        loc='center')
        
        for j, cell in enumerate(table._cells[(0, j)] for j in range(2)):
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#4E79A7')
        
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
        
        plt.title('Performance Metrics Summary', fontsize=16, fontweight='bold')
        
        metrics_path = os.path.join(self.output_dir, f'performance_metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return metrics_path
    
    def plot_comparison(self, results, metrics=None, save_path=None):
        """Plot strategy comparison charts."""
        if metrics is None:
            metrics = ['total_return', 'sharpe_ratio', 'max_drawdown']
        
        strategies = list(results.keys())
        
        def safe_convert_value(value_str):
            if value_str == "N/A" or value_str is None:
                return 0
            if isinstance(value_str, str) and '%' in value_str:
                return float(value_str.strip('%'))
            else:
                return float(value_str)
        
        metric_alternatives = {
            'total_return': ['total_return'],
            'annual_return': ['annual_return'],
            'sharpe_ratio': ['sharpe_ratio'],
            'sortino_ratio': ['sortino_ratio'],
            'max_drawdown': ['max_drawdown'],
            'volatility': ['volatility'],
            'win_rate': ['win_rate'],
            'turnover_rate': ['turnover', 'turnover_rate'],
            'stop_loss_count': ['stop_loss_count'],
            'cost_ratio': ['cost_ratio', 'cost_percentage']
        }
        
        for metric in metrics:
            plt.figure(figsize=(12, 6))
            
            alt_names = metric_alternatives.get(metric, [metric])
            
            values = []
            for strategy in strategies:
                value = None
                for alt_metric in alt_names:
                    if alt_metric in results[strategy]:
                        value = safe_convert_value(results[strategy][alt_metric])
                        break
                
                if value is None:
                    value = 0
                
                values.append(value)
            
            bars = plt.bar(strategies, values, color=[self.colors.get(s, '#333333') for s in strategies])
            
            if values:
                max_value = max(values) if max(values) > 0 else 1
                for bar in bars:
                    height = bar.get_height()
                    if metric in ['total_return', 'annual_return', 'max_drawdown', 'volatility', 'win_rate', 'turnover_rate', 'turnover', 'cost_ratio', 'cost_percentage']:
                        label = f'{height:.1f}%'
                    else:
                        label = f'{height:.2f}'
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.05 * max_value,
                           label, ha='center', va='bottom', fontsize=9)
            
            metric_titles = {
                'total_return': 'Total Return Comparison',
                'annual_return': 'Annual Return Comparison',
                'sharpe_ratio': 'Sharpe Ratio Comparison',
                'sortino_ratio': 'Sortino Ratio Comparison',
                'max_drawdown': 'Maximum Drawdown Comparison',
                'volatility': 'Volatility Comparison',
                'win_rate': 'Win Rate Comparison',
                'turnover_rate': 'Turnover Rate Comparison',
                'turnover': 'Turnover Rate Comparison',
                'stop_loss_count': 'Stop Loss Count Comparison',
                'cost_ratio': 'Cost Ratio Comparison',
                'cost_percentage': 'Cost Ratio Comparison'
            }
            
            plt.title(metric_titles.get(metric, f'{metric.title()} Comparison'), fontsize=14, fontweight='bold')
            plt.xlabel('Strategy', fontsize=12)
            
            y_labels = {
                'total_return': 'Total Return (%)',
                'annual_return': 'Annual Return (%)',
                'sharpe_ratio': 'Sharpe Ratio',
                'sortino_ratio': 'Sortino Ratio',
                'max_drawdown': 'Maximum Drawdown (%)',
                'volatility': 'Volatility (%)',
                'win_rate': 'Win Rate (%)',
                'turnover_rate': 'Turnover Rate (%)',
                'turnover': 'Turnover Rate (%)',
                'stop_loss_count': 'Stop Loss Count',
                'cost_ratio': 'Cost Ratio (%)',
                'cost_percentage': 'Cost Ratio (%)'
            }
            plt.ylabel(y_labels.get(metric, metric.title()), fontsize=12)
            
            plt.xticks(rotation=45, ha='right')
            
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            
            if save_path is None:
                filename = f'{metric}_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
                metric_save_path = os.path.join(self.output_dir, filename)
            else:
                metric_save_path = os.path.join(os.path.dirname(save_path), f'{metric}_{os.path.basename(save_path)}')
            
            plt.savefig(metric_save_path, dpi=300, bbox_inches='tight')
            plt.close()

    def plot_reward_curve(self, rewards, save_path=None):
        """Plot training reward curve."""
        plt.figure(figsize=(10, 6))
        
        plt.gca().set_facecolor(self.colors['background'])
        plt.gcf().set_facecolor('white')
        
        episodes = range(1, len(rewards) + 1)
        plt.plot(episodes, rewards, color=self.colors['main'], linewidth=2, label='Reward')
        
        window = min(50, len(rewards) // 10)
        if window > 0:
            moving_avg = pd.Series(rewards).rolling(window=window).mean()
            plt.plot(episodes, moving_avg, color=self.colors['secondary'], 
                    linewidth=1.5, linestyle='--', label=f'Moving Average (n={window})')
        
        plt.title('Training Reward Curve', pad=15, fontweight='bold')
        plt.xlabel('Episode', labelpad=10)
        plt.ylabel('Reward', labelpad=10)
        
        plt.grid(True, linestyle='--', alpha=0.3, color=self.colors['grid'])
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        
        plt.legend(frameon=True, framealpha=0.9, loc='upper left')
        
        if len(rewards) > 1:
            std = pd.Series(rewards).rolling(window=window).std()
            plt.fill_between(episodes, 
                           moving_avg - std, 
                           moving_avg + std,
                           color=self.colors['main'],
                           alpha=self.fill_alpha)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 
                                   f'reward_curve_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path 

