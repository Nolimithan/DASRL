"""
可视化模块

实现结果可视化功能，包括净值曲线、仓位热力图和绩效指标比较
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
    """可视化器类
    
    提供各种绘图和可视化功能
    """
    
    def __init__(self, output_dir='visualization'):
        """初始化可视化器
        
        Args:
            output_dir (str): 输出目录路径
        """
        self.output_dir = output_dir
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置绘图风格 - 使用更专业的风格
        plt.style.use('fivethirtyeight')
        
        # 设置全局字体为Times New Roman
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['mathtext.fontset'] = 'stix'  # 数学公式字体
        plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
        plt.rcParams['figure.figsize'] = (12, 8)  # 默认图表尺寸
        plt.rcParams['lines.linewidth'] = 2.5  # 线条粗细
        plt.rcParams['axes.labelsize'] = 14  # 轴标签字体大小
        plt.rcParams['axes.titlesize'] = 16  # 标题字体大小
        plt.rcParams['xtick.labelsize'] = 12  # x轴刻度标签字体大小
        plt.rcParams['ytick.labelsize'] = 12  # y轴刻度标签字体大小
        plt.rcParams['legend.fontsize'] = 12  # 图例字体大小
        plt.rcParams['axes.grid'] = True  # 显示网格
        plt.rcParams['grid.alpha'] = 0.3  # 网格透明度
        
        # 对于需要显示中文的文本，在具体绘图函数中单独设置字体
        self.chinese_font = ['SimHei', 'Microsoft YaHei', 'Arial']
        
        # 自定义绘图颜色 - 使用更专业的配色方案
        self.colors = {
            'portfolio': '#0072B2',  # 深蓝色
            'benchmark': '#E69F00',  # 深橙色
            'equal_weight': '#009E73',  # 深绿色
            'mean_var': '#CC79A7',  # 紫红色
            'momentum': '#56B4E9',  # 亮蓝色
            'literature_rl': '#D55E00',  # 深橙红色
            'baseline': '#5A2D81',  # 深紫色
            'no_LIM': '#6C7A89',  # 深灰色
            'fixed_30day': '#F0E442',  # 亮黄色
            'no_cov_penalty': '#0072B2',  # 深蓝色
            'no_exit_rule': '#CC79A7'  # 紫红色
        }
        
        # 设置美化的背景色
        self.bg_color = '#f5f5f5'  # 浅灰色背景
    
    def plot_results(self, portfolio, benchmark=None, save_path=None):
        """绘制策略结果图
        
        Args:
            portfolio (Portfolio): 投资组合实例
            benchmark (pd.DataFrame): 基准数据
            save_path (str): 图表保存路径
        """
        # 获取投资组合历史数据
        history_df = portfolio.get_history_dataframe()
        
        # 创建子图
        fig, axes = plt.subplots(3, 1, figsize=(12, 18), gridspec_kw={'height_ratios': [3, 1, 1]})
        
        # 绘制净值曲线
        self._plot_nav_curve(axes[0], history_df, benchmark)
        
        # 绘制收益率曲线
        self._plot_return_curve(axes[1], history_df)
        
        # 绘制回撤曲线
        self._plot_drawdown_curve(axes[2], history_df)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        if save_path is None:
            save_path = os.path.join(self.output_dir, f'portfolio_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 绘制持仓热力图
        self.plot_holdings_heatmap(portfolio, save_path=None)
        
        # 绘制绩效指标雷达图
        self.plot_performance_radar(portfolio, save_path=None)
        
        # 打印绩效指标
        self._print_performance_metrics(portfolio)
    
    def _plot_nav_curve(self, ax, history_df, benchmark=None):
        """绘制净值曲线
        
        Args:
            ax (matplotlib.axes.Axes): 绘图区域
            history_df (pd.DataFrame): 历史数据
            benchmark (pd.DataFrame): 基准数据
        """
        # 设置背景色
        ax.set_facecolor(self.bg_color)
        
        # 计算净值
        portfolio_nav = history_df['portfolio_value'] / history_df['portfolio_value'].iloc[0]
        
        # 绘制投资组合净值曲线
        ax.plot(history_df['date'], portfolio_nav, 
                label='lim*Enhanced Strategy', 
                color=self.colors['portfolio'], 
                linewidth=3,
                marker='o',
                markersize=0,
                markevery=20)
        
        # 如果有基准数据，绘制基准净值曲线
        if benchmark is not None:
            # 确保基准数据的日期范围与投资组合一致
            benchmark = benchmark[benchmark['date'].isin(history_df['date'])]
            
            if not benchmark.empty:
                # 计算基准净值
                benchmark_nav = benchmark['close'] / benchmark['close'].iloc[0]
                
                # 绘制基准净值曲线
                ax.plot(benchmark['date'], benchmark_nav, 
                       label='Semiconductor ETF Benchmark', 
                       color=self.colors['benchmark'], 
                       linewidth=2.5, 
                       linestyle='--',
                       marker='s',
                       markersize=0,
                       markevery=20)
        
        # 设置标题和标签 - 使用更专业的样式
        ax.set_title('Net Value Comparison', fontsize=16, fontweight='bold', pad=15)
        ax.set_xlabel('Date', fontsize=14, labelpad=10)
        ax.set_ylabel('Net Value (Initial=1)', fontsize=14, labelpad=10)
        
        # 格式化y轴为百分比
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.2f}'))
        
        # 添加网格线
        ax.grid(True, linestyle='--', alpha=0.3, color='gray')
        
        # 添加图例
        ax.legend(loc='upper left', fontsize=12, frameon=True, framealpha=0.9)
        
        # 添加阴影区域以突出策略超额收益
        if benchmark is not None and not benchmark.empty:
            # 计算策略超额收益的区域
            ax.fill_between(history_df['date'], 
                           portfolio_nav, 
                           benchmark_nav, 
                           where=(portfolio_nav > benchmark_nav),
                           color=self.colors['portfolio'], 
                           alpha=0.15,
                           interpolate=True)
        
        # 设置x轴刻度格式
        ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mpl.dates.MonthLocator(interval=3))
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # 突出显示重要日期
        ax.axhline(y=1.0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    
    def _plot_return_curve(self, ax, history_df):
        """绘制收益率曲线
        
        Args:
            ax (matplotlib.axes.Axes): 绘图区域
            history_df (pd.DataFrame): 历史数据
        """
        # 设置背景色
        ax.set_facecolor(self.bg_color)
        
        # 计算累积收益率
        cum_return = (history_df['portfolio_value'] / history_df['portfolio_value'].iloc[0]) - 1
        
        # 绘制累积收益率曲线
        ax.plot(history_df['date'], cum_return * 100, 
               label='Cumulative Return', 
               color=self.colors['portfolio'], 
               linewidth=3)
        
        # 设置标题和标签
        ax.set_title('Cumulative Return', fontsize=16, fontweight='bold', pad=15)
        ax.set_xlabel('Date', fontsize=14, labelpad=10)
        ax.set_ylabel('Return (%)', fontsize=14, labelpad=10)
        
        # 添加水平线表示零收益
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        
        # 格式化y轴为百分比
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0f}%'))
        
        # 添加网格线
        ax.grid(True, linestyle='--', alpha=0.3, color='gray')
        
        # 添加图例
        ax.legend(loc='upper left', fontsize=12, frameon=True, framealpha=0.9)
        
        # 填充收益率曲线下方区域
        ax.fill_between(history_df['date'], 0, cum_return * 100, 
                      color=self.colors['portfolio'], alpha=0.15)
        
        # 设置x轴刻度格式
        ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mpl.dates.MonthLocator(interval=3))
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    def _plot_drawdown_curve(self, ax, history_df):
        """绘制回撤曲线
        
        Args:
            ax (matplotlib.axes.Axes): 绘图区域
            history_df (pd.DataFrame): 历史数据
        """
        # 设置背景色
        ax.set_facecolor(self.bg_color)
        
        # 计算滚动最大值
        rolling_max = history_df['portfolio_value'].cummax()
        
        # 计算回撤
        drawdown = (rolling_max - history_df['portfolio_value']) / rolling_max
        
        # 绘制回撤曲线
        ax.fill_between(history_df['date'], 0, drawdown * 100, 
                      alpha=0.3, 
                      color='#D55E00',  # 使用深橙红色表示回撤
                      label='Drawdown')
        
        ax.plot(history_df['date'], drawdown * 100, 
               color=self.colors['portfolio'], 
               linewidth=1)
        
        # 设置标题和标签
        ax.set_title('Drawdown Curve', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Drawdown (%)', fontsize=12)
        
        # 格式化y轴为百分比
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0f}%'))
        
        # 反转y轴，使回撤向下显示
        ax.invert_yaxis()
        
        # 添加网格线
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 添加图例
        ax.legend(loc='lower left', fontsize=10)
    
    def plot_holdings_heatmap(self, portfolio, save_path=None):
        """绘制持仓热力图
        
        Args:
            portfolio (Portfolio): 投资组合实例
            save_path (str): 图表保存路径
        """
        # 提取持仓历史
        holdings_history = portfolio.history['holdings']
        dates = portfolio.history['date']
        
        # 将持仓历史转换为适合热力图的格式
        # 首先找出所有唯一的股票代码
        all_symbols = set()
        for holdings in holdings_history:
            all_symbols.update(holdings)
        all_symbols = sorted(list(all_symbols))
        
        # 创建权重矩阵
        weights_matrix = np.zeros((len(dates), len(all_symbols)))
        
        # 填充权重矩阵
        for i, (date, holdings) in enumerate(zip(dates, holdings_history)):
            # 获取当日权重
            weights = portfolio.history['transactions'][i]
            for j, symbol in enumerate(all_symbols):
                if symbol in weights:
                    weights_matrix[i, j] = weights[symbol]
        
        # 创建DataFrame
        weights_df = pd.DataFrame(weights_matrix, index=dates, columns=all_symbols)
        
        # 创建热力图
        plt.figure(figsize=(12, 10))
        sns.heatmap(weights_df, cmap='YlGnBu', linewidths=0.5, 
                   cbar_kws={'label': 'Portfolio Weight'})
        
        # 设置标题和标签
        plt.title('Holdings Heatmap', fontsize=14, fontweight='bold')
        plt.xlabel('Stocks', fontsize=12)
        plt.ylabel('Date', fontsize=12)
        
        # 旋转x轴标签以便更好地显示
        plt.xticks(rotation=45, ha='right')
        
        # 保存图表
        if save_path is None:
            save_path = os.path.join(self.output_dir, f'holdings_heatmap_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_performance_radar(self, portfolio, save_path=None):
        """绘制绩效指标雷达图
        
        Args:
            portfolio (Portfolio): 投资组合实例
            save_path (str): 图表保存路径
        """
        # 获取绩效指标
        metrics = portfolio.get_performance_summary()
        
        # 选择要在雷达图中显示的指标
        # 注意：雷达图需要所有值为正数，因此最大回撤需要取正值
        radar_metrics = {}
        
        # 安全地处理各种可能的指标值格式
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
            'Max Drawdown': -safe_convert_percent(metrics['max_drawdown']),  # 取负值使得较小的回撤显示为较好的性能
            'Volatility': -safe_convert_percent(metrics['volatility'])  # 取负值使得较小的波动率显示为较好的性能
        }
        
        # 调整指标范围
        # 标准化到0-1范围
        min_val = min(radar_metrics.values())
        max_val = max(radar_metrics.values())
        
        for key in radar_metrics:
            # 将值标准化到0-1
            radar_metrics[key] = (radar_metrics[key] - min_val) / (max_val - min_val)
        
        # 准备雷达图数据
        categories = list(radar_metrics.keys())
        values = list(radar_metrics.values())
        
        # 添加首尾相连
        values.append(values[0])
        categories.append(categories[0])
        
        # 计算角度
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=True)
        
        # 创建雷达图
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
        
        # 填充雷达图
        ax.fill(angles, values, color=self.colors['portfolio'], alpha=0.25)
        ax.plot(angles, values, color=self.colors['portfolio'], linewidth=2)
        
        # 添加指标标签
        ax.set_thetagrids(angles * 180 / np.pi, categories, fontsize=12)
        
        # 设置标题
        ax.set_title('Performance Metrics', fontsize=14, fontweight='bold')
        
        # 添加径向网格线
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
        ax.set_ylim(0, 1)
        
        # 保存图表
        if save_path is None:
            save_path = os.path.join(self.output_dir, f'performance_radar_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _print_performance_metrics(self, portfolio):
        """打印绩效指标
        
        Args:
            portfolio (Portfolio): 投资组合实例
        """
        metrics = portfolio.get_performance_summary()
        
        # 检查获取的指标键名，确保使用正确的键名
        print("获取到的指标键名:", list(metrics.keys()))
        
        # 安全地获取指标值的辅助函数
        def safe_get_metric(key, alternative_keys=None):
            if key in metrics:
                return metrics[key]
            elif alternative_keys:
                # 尝试其他可能的键名
                for alt_key in alternative_keys:
                    if alt_key in metrics:
                        return metrics[alt_key]
            # 如果都找不到，返回默认值
            return "N/A"
        
        # 安全地格式化指标值
        def format_metric(key, value):
            # 确保所有值都是字符串
            if isinstance(value, (int, float, np.number)):
                if key in ['total_return', 'annual_return', 'max_drawdown', 'volatility', 'win_rate', 'turnover', 'turnover_rate', 'cost_ratio', 'cost_percentage']:
                    # 这些指标应显示为百分比
                    return f"{float(value):.2f}%"
                else:
                    # 其他数值指标
                    return f"{float(value):.4f}"
            return str(value)
        
        # 创建绩效指标表格
        plt.figure(figsize=(12, 8))
        
        # 创建一个没有边框的表格
        ax = plt.subplot(111, frame_on=False)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        
        # 表格数据 - 使用safe_get_metric确保不会出现KeyError
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
        
        # 创建表格
        table = ax.table(cellText=metrics_data,
                        colWidths=[0.3, 0.2],
                        cellLoc='center',
                        loc='center')
        
        # 设置标题行样式
        for j, cell in enumerate(table._cells[(0, j)] for j in range(2)):
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#4E79A7')
        
        # 设置表格样式
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
        
        plt.title('Performance Metrics Summary', fontsize=16, fontweight='bold')
        
        # 保存图表
        metrics_path = os.path.join(self.output_dir, f'performance_metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return metrics_path
    
    def plot_comparison(self, results, metrics=None, save_path=None):
        """绘制策略比较图
        
        Args:
            results (dict): 不同策略的结果
            metrics (list): 要对比的指标列表
            save_path (str): 图表保存路径
        """
        if metrics is None:
            metrics = ['total_return', 'sharpe_ratio', 'max_drawdown']
        
        # 提取策略名称
        strategies = list(results.keys())
        
        # 安全地转换百分比值为浮点数
        def safe_convert_value(value_str):
            if value_str == "N/A" or value_str is None:
                return 0
            if isinstance(value_str, str) and '%' in value_str:
                return float(value_str.strip('%'))
            else:
                return float(value_str)
        
        # 指标名称到可能的替代名称的映射
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
        
        # 为每个指标创建一个对比图
        for metric in metrics:
            plt.figure(figsize=(12, 6))
            
            # 找到这个指标可能的替代名称
            alt_names = metric_alternatives.get(metric, [metric])
            
            # 收集所有策略的指标值
            values = []
            for strategy in strategies:
                # 从策略结果中找到正确的指标键
                value = None
                for alt_metric in alt_names:
                    if alt_metric in results[strategy]:
                        value = safe_convert_value(results[strategy][alt_metric])
                        break
                
                # 如果所有替代名称都没找到，设为0
                if value is None:
                    value = 0
                
                values.append(value)
            
            # 创建条形图
            bars = plt.bar(strategies, values, color=[self.colors.get(s, '#333333') for s in strategies])
            
            # 添加数值标签
            if values:  # 确保values不为空
                max_value = max(values) if max(values) > 0 else 1  # 避免除零错误
                for bar in bars:
                    height = bar.get_height()
                    if metric in ['total_return', 'annual_return', 'max_drawdown', 'volatility', 'win_rate', 'turnover_rate', 'turnover', 'cost_ratio', 'cost_percentage']:
                        # 这些指标用百分比表示
                        label = f'{height:.1f}%'
                    else:
                        label = f'{height:.2f}'
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.05 * max_value,
                           label, ha='center', va='bottom', fontsize=9)
            
            # 为每个指标设置合适的标题
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
            
            # 设置标题和标签
            plt.title(metric_titles.get(metric, f'{metric.title()} Comparison'), fontsize=14, fontweight='bold')
            plt.xlabel('Strategy', fontsize=12)
            
            # 为每个指标设置合适的y轴标签
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
            
            # 旋转x轴标签以避免重叠
            plt.xticks(rotation=45, ha='right')
            
            # 添加网格线
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # 调整布局
            plt.tight_layout()
            
            # 保存图表
            if save_path is None:
                filename = f'{metric}_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
                metric_save_path = os.path.join(self.output_dir, filename)
            else:
                # 如果提供了基本路径，在其中添加指标名称
                metric_save_path = os.path.join(os.path.dirname(save_path), f'{metric}_{os.path.basename(save_path)}')
            
            plt.savefig(metric_save_path, dpi=300, bbox_inches='tight')
            plt.close() 

