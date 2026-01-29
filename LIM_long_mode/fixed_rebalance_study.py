import os
import numpy as np
import pandas as pd
import yaml
from datetime import datetime
from utils.environment import TradingEnvironment
from models.ppo_agent import PPOAgent as PPOModel
from utils.data_loader import DataLoader
from utils.lim_calculator import LIMCalculator
import argparse
import time
from utils.portfolio import Portfolio
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Run fixed rebalance period study')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                      help='Run mode: train or test')
    parser.add_argument('--config', type=str, default='config/default.yaml',
                      help='Config file path')
    parser.add_argument('--start_date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--group_type', type=str, default='TOP',
                      choices=['TOP', 'MIDDLE', 'LOW'],
                      help='Stock group type')
    parser.add_argument('--runs', type=int, default=5,
                      help='Runs per strategy (for statistical analysis)')
    return parser.parse_args()

def load_config(config_path):
    """Load configuration file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

class FixedRebalanceStudy:
    """Fixed rebalance period study."""
    
    def __init__(self, config, mode='train', start_date=None, end_date=None, runs=5):
        """Initialize the fixed rebalance study.

        Args:
            config (dict): Configuration dictionary.
            mode (str): Run mode ('train' or 'test').
            start_date (str): Start date.
            end_date (str): End date.
            runs (int): Runs per strategy.
        """
        self.config = config
        self.mode = mode
        self.runs = runs
        
        # Update date range if provided.
        if start_date:
            self.config['data']['start_date'] = start_date
        if end_date:
            self.config['data']['end_date'] = end_date
            
        # Create output directory.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(config['paths']['results_path'], f'fixed_rebalance_study_{timestamp}')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Ensure data directory exists.
        data_dir = os.path.join(os.getcwd(), 'data')
        os.makedirs(data_dir, exist_ok=True)
        
        print("Initializing fixed rebalance period study...")
        print(f"Date range: {self.config['data']['start_date']} to {self.config['data']['end_date']}")
        print(f"Runs per strategy: {self.runs}")
        
        try:
            # Initialize data loader.
            self.data_loader = DataLoader(
                start_date=self.config['data']['start_date'],
                end_date=self.config['data']['end_date'],
                symbols=self.config['data']['symbols'],
                data_dir=data_dir
            )
            
            # Load data.
            print("Loading data...")
            self.data_loader.load_data()
            
            # Validate data load.
            if not self.data_loader.data.empty and not self.data_loader.benchmark.empty:
                loaded_symbols = len(self.data_loader.all_stock_data)
                total_symbols = len(self.config['data']['symbols'])
                print(f"Loaded data for {loaded_symbols}/{total_symbols} symbols")

                # Update symbols list with successfully loaded ones.
                if loaded_symbols < total_symbols:
                    print(f"Warning: some symbols failed to load; continuing with {loaded_symbols} symbols")
                    self.config['data']['symbols'] = list(self.data_loader.all_stock_data.keys())
            else:
                raise ValueError("Data load failed. Check data source and parameters.")
                
        except Exception as e:
            print(f"Data load error: {str(e)}")
            raise
        
        # Initialize LIM calculator.
        self.lim_calculator = LIMCalculator(
            window_size=self.config['lim']['window_size'],
            alpha=self.config['lim']['alpha'],
            lookback_days=self.config['lim']['lookback_days']
        )
        
        # Save experiment config.
        self._save_config()
        
        # Define rebalance periods (10 to 90 days, step 5).
        self.rebalance_periods = list(range(10, 95, 5))
        print(f"Rebalance periods to test: {self.rebalance_periods}")
    
    def _save_config(self):
        """Save experiment configuration."""
        config_path = os.path.join(self.output_dir, 'experiment_config.yaml')
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
    
    def run_fixed_period_strategy(self, rebalance_period, run_id=0):
        """Run fixed-period rebalance strategy.

        Args:
            rebalance_period (int): Rebalance period in days.
            run_id (int): Run ID for random seed.

        Returns:
            dict: Strategy results.
        """
        print(f"Running fixed {rebalance_period}-day rebalance (run {run_id+1})...")
        
        # Set random seed for reproducibility.
        np.random.seed(42 + run_id)
        
        # Create config copy.
        config = self.config.copy()
        config['environment']['rebalance_period'] = rebalance_period
        config['environment']['dynamic_rebalance'] = False
        config['environment']['random_rebalance'] = False
        
        env = TradingEnvironment(
            config=config,
            data_loader=self.data_loader,
            lim_calculator=self.lim_calculator
        )
        
        model = PPOModel(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            lr=config['rl']['learning_rate'],
            gamma=config['rl']['gamma'],
            clip_ratio=config['rl']['clip_ratio'],
            value_coef=config['rl']['value_coef'],
            entropy_coef=config['rl']['entropy_coef']
        )
        
        if self.mode == 'train':
            model.train(env, epochs=config['rl']['epochs'])
        else:
            model.load(config['paths']['model_load_path'])
            
        portfolio = Portfolio(
            initial_capital=config['portfolio']['initial_capital'],
            commission_rate=config['trading']['commission_rate']
        )
        
        env.reset()
        portfolio.run_strategy(model, env)
        
        results = portfolio.get_performance_summary()
        
        results['dates'] = portfolio.history['date']
        results['portfolio_values'] = portfolio.history['portfolio_value']
        results['rebalance_period'] = rebalance_period
        results['run_id'] = run_id
        
        return results
    
    def run_all_fixed_period_experiments(self):
        """Run all fixed-period rebalance experiments."""
        print("Starting all fixed-period rebalance experiments...")
        
        all_results = {}
        
        for period in self.rebalance_periods:
            period_results = []
            
            for run_id in range(self.runs):
                try:
                    result = self.run_fixed_period_strategy(period, run_id)
                    period_results.append(result)
                    
                    print(f"Period {period} days - run {run_id+1}: total return={result.get('total_return', 0):.2%}, "
                          f"Sharpe={result.get('sharpe_ratio', 0):.3f}")
                    
                except Exception as e:
                    print(f"Error in period {period} days run {run_id+1}: {e}")
                    continue
            
            all_results[period] = period_results
            print(f"Completed {len(period_results)} runs for {period}-day period\n")
        
        self._save_raw_results(all_results)
        
        self._analyze_and_visualize_results(all_results)
        
        return all_results
    
    def _save_raw_results(self, all_results):
        """Save raw experiment results."""
        print("Saving raw experiment results...")
        
        summary_data = []
        
        for period, period_results in all_results.items():
            for result in period_results:
                summary_row = {
                    'rebalance_period': period,
                    'run_id': result.get('run_id', 0),
                    'total_return': result.get('total_return', 0),
                    'sharpe_ratio': result.get('sharpe_ratio', 0),
                    'max_drawdown': result.get('max_drawdown', 0),
                    'win_rate': result.get('win_rate', 0),
                    'volatility': result.get('volatility', 0),
                    'sortino_ratio': result.get('sortino_ratio', 0),
                    'turnover_rate': result.get('turnover_rate', 0),
                    'final_value': result.get('final_value', 0)
                }
                summary_data.append(summary_row)
        
        df_summary = pd.DataFrame(summary_data)
        summary_path = os.path.join(self.output_dir, 'fixed_rebalance_summary.csv')
        df_summary.to_csv(summary_path, index=False)
        print(f"Summary results saved to: {summary_path}")
        
        portfolio_data = []
        
        for period, period_results in all_results.items():
            for result in period_results:
                if 'dates' in result and 'portfolio_values' in result:
                    for date, value in zip(result['dates'], result['portfolio_values']):
                        portfolio_data.append({
                            'rebalance_period': period,
                            'run_id': result.get('run_id', 0),
                            'date': date,
                            'portfolio_value': value
                        })
        
        df_portfolio = pd.DataFrame(portfolio_data)
        portfolio_path = os.path.join(self.output_dir, 'fixed_rebalance_portfolio_values.csv')
        df_portfolio.to_csv(portfolio_path, index=False)
        print(f"Portfolio value data saved to: {portfolio_path}")
    
    def _analyze_and_visualize_results(self, all_results):
        """Analyze and visualize results."""
        print("Starting analysis and visualization...")
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = 'Times New Roman'
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.unicode_minus'] = False
        
        tr_data = {}
        for period, period_results in all_results.items():
            tr_values = [result.get('total_return', 0) for result in period_results]
            tr_data[period] = tr_values
        
        self._plot_total_return_distribution(tr_data)
        
        self._plot_return_comparison_with_confidence(tr_data)
        
        self._plot_normality_tests(tr_data)
        
        self._plot_tail_analysis(tr_data)
        
        self._generate_statistical_report(tr_data)
    
    def _plot_total_return_distribution(self, tr_data):
        """Plot total return distributions."""
        print("Plotting total return distributions...")
        
        n_periods = len(tr_data)
        cols = 4
        rows = (n_periods + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(20, 5*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        for i, (period, returns) in enumerate(tr_data.items()):
            ax = axes[i]
            
            ax.hist(returns, bins=max(3, len(returns)//2), alpha=0.7, color='skyblue', 
                   density=True, label=f'{period}-day rebalance')
            
            mu, sigma = stats.norm.fit(returns)
            x = np.linspace(min(returns), max(returns), 100)
            normal_pdf = stats.norm.pdf(x, mu, sigma)
            ax.plot(x, normal_pdf, 'r-', linewidth=2, label=f'Normal Dist (μ={mu:.3f}, σ={sigma:.3f})')
            
            if period == 30:
                ax.set_facecolor('#f0f8ff')
                ax.set_title(f'{period}-day rebalance (primary strategy)', fontweight='bold', color='red')
            else:
                ax.set_title(f'{period}-day rebalance')
            
            ax.set_xlabel('Total Return')
            ax.set_ylabel('Probability Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            skewness = stats.skew(returns)
            kurtosis = stats.kurtosis(returns)
            ax.text(0.05, 0.95, f'Skewness: {skewness:.3f}\nKurtosis: {kurtosis:.3f}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        for i in range(n_periods, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        distribution_path = os.path.join(self.output_dir, 'total_return_distributions.png')
        plt.savefig(distribution_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Total return distribution plot saved to: {distribution_path}")
    
    def _plot_return_comparison_with_confidence(self, tr_data):
        """Plot total return comparison with confidence intervals."""
        print("Plotting total return comparison with confidence intervals...")
        
        periods = sorted(tr_data.keys())
        means = []
        stds = []
        medians = []
        
        for period in periods:
            returns = tr_data[period]
            means.append(np.mean(returns))
            stds.append(np.std(returns))
            medians.append(np.median(returns))
        
        means = np.array(means)
        stds = np.array(stds)
        medians = np.array(medians)
        
        confidence_level = 0.95
        alpha = 1 - confidence_level
        df_freedom = len(tr_data[periods[0]]) - 1
        t_critical = stats.t.ppf(1 - alpha/2, df_freedom)
        
        se = stds / np.sqrt(len(tr_data[periods[0]]))
        ci_lower = means - t_critical * se
        ci_upper = means + t_critical * se
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        ax1.plot(periods, means, 'b-', linewidth=2, marker='o', markersize=6, label='Mean')
        ax1.fill_between(periods, ci_lower, ci_upper, alpha=0.3, color='lightblue', 
                        label=f'{confidence_level*100:.0f}% Confidence Interval')
        
        idx_30 = periods.index(30) if 30 in periods else None
        if idx_30 is not None:
            ax1.scatter([30], [means[idx_30]], color='red', s=100, zorder=5, 
                       label='Primary strategy (30-day)')
            ax1.annotate(f'Primary: {means[idx_30]:.3f}', 
                        xy=(30, means[idx_30]), xytext=(35, means[idx_30]+0.01),
                        arrowprops=dict(arrowstyle='->', color='red'),
                        fontsize=10, color='red', fontweight='bold')
        
        ax1.set_xlabel('Rebalance Period (Days)')
        ax1.set_ylabel('Mean Total Return')
        ax1.set_title('Mean Total Return by Rebalance Period (with CI)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        q1_values = []
        q3_values = []
        
        for period in periods:
            returns = tr_data[period]
            q1_values.append(np.percentile(returns, 25))
            q3_values.append(np.percentile(returns, 75))
        
        q1_values = np.array(q1_values)
        q3_values = np.array(q3_values)
        
        ax2.plot(periods, medians, 'g-', linewidth=2, marker='s', markersize=6, label='Median')
        ax2.fill_between(periods, q1_values, q3_values, alpha=0.3, color='lightgreen', 
                        label='Interquartile Range (Q1–Q3)')
        
        if idx_30 is not None:
            ax2.scatter([30], [medians[idx_30]], color='red', s=100, zorder=5)
            ax2.annotate(f'Primary: {medians[idx_30]:.3f}', 
                        xy=(30, medians[idx_30]), xytext=(35, medians[idx_30]+0.01),
                        arrowprops=dict(arrowstyle='->', color='red'),
                        fontsize=10, color='red', fontweight='bold')
        
        ax2.set_xlabel('Rebalance Period (Days)')
        ax2.set_ylabel('Median Total Return')
        ax2.set_title('Median Total Return by Rebalance Period (with IQR)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        comparison_path = os.path.join(self.output_dir, 'return_comparison_with_confidence.png')
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Total return comparison plot saved to: {comparison_path}")
    
    def _plot_normality_tests(self, tr_data):
        """Plot normality test results."""
        print("Running normality tests...")
        
        periods = sorted(tr_data.keys())
        shapiro_stats = []
        shapiro_pvalues = []
        jarque_bera_stats = []
        jarque_bera_pvalues = []
        
        for period in periods:
            returns = tr_data[period]
            
            if len(returns) >= 3:
                shapiro_stat, shapiro_p = stats.shapiro(returns)
                shapiro_stats.append(shapiro_stat)
                shapiro_pvalues.append(shapiro_p)
            else:
                shapiro_stats.append(np.nan)
                shapiro_pvalues.append(np.nan)
            
            if len(returns) >= 8:
                jb_stat, jb_p = stats.jarque_bera(returns)
                jarque_bera_stats.append(jb_stat)
                jarque_bera_pvalues.append(jb_p)
            else:
                jarque_bera_stats.append(np.nan)
                jarque_bera_pvalues.append(np.nan)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        colors = ['red' if p < 0.05 else 'green' for p in shapiro_pvalues]
        bars1 = ax1.bar(range(len(periods)), shapiro_pvalues, color=colors, alpha=0.7)
        ax1.axhline(y=0.05, color='red', linestyle='--', label='Significance (α=0.05)')
        ax1.set_xlabel('Rebalance Period (Days)')
        ax1.set_ylabel('p-value')
        ax1.set_title('Shapiro-Wilk Normality Test\n(Red: reject, Green: accept)')
        ax1.set_xticks(range(len(periods)))
        ax1.set_xticklabels([str(p) for p in periods], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        idx_30 = periods.index(30) if 30 in periods else None
        if idx_30 is not None:
            ax1.annotate('Primary strategy', xy=(idx_30, shapiro_pvalues[idx_30]), 
                        xytext=(idx_30, shapiro_pvalues[idx_30]+0.1),
                        arrowprops=dict(arrowstyle='->', color='blue'),
                        fontsize=10, color='blue', fontweight='bold')
        
        colors = ['red' if p < 0.05 else 'green' for p in jarque_bera_pvalues if not np.isnan(p)]
        valid_jb_pvalues = [p for p in jarque_bera_pvalues if not np.isnan(p)]
        valid_periods_idx = [i for i, p in enumerate(jarque_bera_pvalues) if not np.isnan(p)]
        
        if valid_jb_pvalues:
            bars2 = ax2.bar(valid_periods_idx, valid_jb_pvalues, color=colors, alpha=0.7)
            ax2.axhline(y=0.05, color='red', linestyle='--', label='Significance (α=0.05)')
            ax2.set_xlabel('Rebalance Period (Days)')
            ax2.set_ylabel('p-value')
            ax2.set_title('Jarque-Bera Normality Test\n(Red: reject, Green: accept)')
            ax2.set_xticks(valid_periods_idx)
            ax2.set_xticklabels([str(periods[i]) for i in valid_periods_idx], rotation=45)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            if idx_30 is not None and not np.isnan(jarque_bera_pvalues[idx_30]):
                jb_idx_30 = valid_periods_idx.index(idx_30)
                ax2.annotate('Primary strategy', xy=(idx_30, jarque_bera_pvalues[idx_30]), 
                            xytext=(idx_30, jarque_bera_pvalues[idx_30]+0.1),
                            arrowprops=dict(arrowstyle='->', color='blue'),
                            fontsize=10, color='blue', fontweight='bold')
        
        plt.tight_layout()
        normality_path = os.path.join(self.output_dir, 'normality_tests.png')
        plt.savefig(normality_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Normality test plot saved to: {normality_path}")
    
    def _plot_tail_analysis(self, tr_data):
        """Plot tail analysis."""
        print("Analyzing tail characteristics...")
        
        periods = sorted(tr_data.keys())
        skewness_values = []
        kurtosis_values = []
        
        for period in periods:
            returns = tr_data[period]
            skewness_values.append(stats.skew(returns))
            kurtosis_values.append(stats.kurtosis(returns))
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        colors1 = ['red' if abs(s) > 0.5 else 'blue' for s in skewness_values]
        bars1 = ax1.bar(range(len(periods)), skewness_values, color=colors1, alpha=0.7)
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Skewness Threshold (±0.5)')
        ax1.axhline(y=-0.5, color='red', linestyle='--', alpha=0.7)
        ax1.set_xlabel('Rebalance Period (Days)')
        ax1.set_ylabel('Skewness')
        ax1.set_title('Return Skewness\n(Red: significant skew)')
        ax1.set_xticks(range(len(periods)))
        ax1.set_xticklabels([str(p) for p in periods], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        idx_30 = periods.index(30) if 30 in periods else None
        if idx_30 is not None:
            ax1.annotate('Primary', xy=(idx_30, skewness_values[idx_30]), 
                        xytext=(idx_30, skewness_values[idx_30]+0.2),
                        arrowprops=dict(arrowstyle='->', color='green'),
                        fontsize=10, color='green', fontweight='bold')
        
        colors2 = ['red' if k > 3 else 'blue' for k in kurtosis_values]
        bars2 = ax2.bar(range(len(periods)), kurtosis_values, color=colors2, alpha=0.7)
        ax2.axhline(y=3, color='red', linestyle='--', alpha=0.7, label='Normal Kurtosis (3)')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax2.set_xlabel('Rebalance Period (Days)')
        ax2.set_ylabel('Kurtosis')
        ax2.set_title('Return Kurtosis\n(Red: high kurtosis)')
        ax2.set_xticks(range(len(periods)))
        ax2.set_xticklabels([str(p) for p in periods], rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        if idx_30 is not None:
            ax2.annotate('Primary', xy=(idx_30, kurtosis_values[idx_30]), 
                        xytext=(idx_30, kurtosis_values[idx_30]+0.5),
                        arrowprops=dict(arrowstyle='->', color='green'),
                        fontsize=10, color='green', fontweight='bold')
        
        colors3 = ['red' if period == 30 else 'blue' for period in periods]
        sizes = [100 if period == 30 else 50 for period in periods]
        scatter = ax3.scatter(skewness_values, kurtosis_values, c=colors3, s=sizes, alpha=0.7)
        
        ax3.axhline(y=3, color='gray', linestyle='--', alpha=0.5, label='Normal Kurtosis')
        ax3.axvline(x=0, color='gray', linestyle='--', alpha=0.5, label='Normal Skewness')
        
        for i, period in enumerate(periods):
            if period == 30:
                ax3.annotate(f'{period}-day\n(Primary)', 
                           (skewness_values[i], kurtosis_values[i]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=10, color='red', fontweight='bold')
            else:
                ax3.annotate(f'{period}', 
                           (skewness_values[i], kurtosis_values[i]),
                           xytext=(3, 3), textcoords='offset points',
                           fontsize=8)
        
        ax3.set_xlabel('Skewness')
        ax3.set_ylabel('Kurtosis')
        ax3.set_title('Skewness vs Kurtosis\n(Red: primary strategy)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        tail_analysis_path = os.path.join(self.output_dir, 'tail_analysis.png')
        plt.savefig(tail_analysis_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Tail analysis plot saved to: {tail_analysis_path}")
    
    def _generate_statistical_report(self, tr_data):
        """Generate statistical analysis report."""
        print("Generating statistical analysis report...")
        
        report_lines = []
        report_lines.append("# Fixed Rebalance Period Study Report")
        report_lines.append("## Overview")
        report_lines.append(f"- Tested period range: {min(tr_data.keys())} to {max(tr_data.keys())} days")
        report_lines.append(f"- Runs per period: {self.runs}")
        report_lines.append("- Primary strategy: 30-day rebalance")
        report_lines.append("")
        
        report_lines.append("## Summary by Rebalance Period")
        report_lines.append("| Period (Days) | Mean | Std Dev | Skewness | Kurtosis | Min | Max | Shapiro p-value |")
        report_lines.append("|---------|------|--------|------|------|--------|--------|-------------|")
        
        for period in sorted(tr_data.keys()):
            returns = tr_data[period]
            mean_val = np.mean(returns)
            std_val = np.std(returns)
            skew_val = stats.skew(returns)
            kurt_val = stats.kurtosis(returns)
            min_val = np.min(returns)
            max_val = np.max(returns)
            
            if len(returns) >= 3:
                _, shapiro_p = stats.shapiro(returns)
            else:
                shapiro_p = np.nan
            
            period_str = f"**{period}**" if period == 30 else str(period)
            
            report_lines.append(f"| {period_str} | {mean_val:.4f} | {std_val:.4f} | {skew_val:.3f} | {kurt_val:.3f} | {min_val:.4f} | {max_val:.4f} | {shapiro_p:.3f} |")
        
        report_lines.append("")
        
        report_lines.append("## Normality Test Summary")
        normal_periods = []
        non_normal_periods = []
        
        for period in sorted(tr_data.keys()):
            returns = tr_data[period]
            if len(returns) >= 3:
                _, shapiro_p = stats.shapiro(returns)
                if shapiro_p > 0.05:
                    normal_periods.append(period)
                else:
                    non_normal_periods.append(period)
        
        report_lines.append(f"- Periods consistent with normality (p>0.05): {normal_periods}")
        report_lines.append(f"- Periods rejecting normality (p≤0.05): {non_normal_periods}")
        report_lines.append("")
        
        report_lines.append("## Tail Characteristics Analysis")
        
        if 30 in tr_data:
            returns_30 = tr_data[30]
            skew_30 = stats.skew(returns_30)
            kurt_30 = stats.kurtosis(returns_30)
            
            report_lines.append("### Primary Strategy (30-day rebalance)")
            report_lines.append(f"- Skewness: {skew_30:.3f} {'(significant skew)' if abs(skew_30) > 0.5 else '(near symmetric)'}")
            report_lines.append(f"- Kurtosis: {kurt_30:.3f} {'(fat-tailed)' if kurt_30 > 3 else '(light-tailed)'}")
            
            if abs(skew_30) > 0.5 or kurt_30 > 3:
                report_lines.append("- **Conclusion**: 30-day strategy returns show fat-tail characteristics")
            else:
                report_lines.append("- **Conclusion**: 30-day strategy returns are close to normal")
        
        report_lines.append("")
        
        fat_tail_count = 0
        for period in tr_data.keys():
            returns = tr_data[period]
            skew_val = stats.skew(returns)
            kurt_val = stats.kurtosis(returns)
            if abs(skew_val) > 0.5 or kurt_val > 3:
                fat_tail_count += 1
        
        report_lines.append("### Overall Analysis")
        report_lines.append(f"- Periods with fat-tail characteristics: {fat_tail_count}/{len(tr_data)}")
        report_lines.append(f"- Fat-tail ratio: {fat_tail_count/len(tr_data)*100:.1f}%")
        
        if fat_tail_count / len(tr_data) > 0.5:
            report_lines.append("- **Conclusion**: Most periods show fat-tail characteristics")
        else:
            report_lines.append("- **Conclusion**: Most periods are close to normal")
        
        report_path = os.path.join(self.output_dir, 'statistical_analysis_report.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"Statistical analysis report saved to: {report_path}")
        
        print("\n=== Key Findings ===")
        if 30 in tr_data:
            returns_30 = tr_data[30]
            skew_30 = stats.skew(returns_30)
            kurt_30 = stats.kurtosis(returns_30)
            print(f"30-day strategy - skewness: {skew_30:.3f}, kurtosis: {kurt_30:.3f}")
            if abs(skew_30) > 0.5 or kurt_30 > 3:
                print("✓ 30-day strategy returns show fat-tail characteristics")
            else:
                print("✗ 30-day strategy returns are close to normal")
        
        print(f"Overall: {fat_tail_count}/{len(tr_data)} periods show fat-tail characteristics ({fat_tail_count/len(tr_data)*100:.1f}%)")

if __name__ == '__main__':
    args = parse_args()
    
    config = load_config(args.config)
    
    study = FixedRebalanceStudy(
        config=config,
        mode=args.mode,
        start_date=args.start_date,
        end_date=args.end_date,
        runs=args.runs
    )
    
    config['environment']['group_type'] = args.group_type
    
    all_results = study.run_all_fixed_period_experiments()
    
    print("\n=== Fixed rebalance period study complete ===")
    print(f"Results saved to: {study.output_dir}") 

