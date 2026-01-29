"""
Visualize LIM time-series curves for 30 stocks over 100 days.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import yaml
from utils.data_loader import DataLoader
from utils.lim_calculator import LIMCalculator
import warnings
warnings.filterwarnings("ignore")

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10


class LIMCurvesVisualizer:
    """LIM curve visualizer."""
    
    def __init__(self, config_path='config/default.yaml'):
        """Initialize the visualizer."""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.data_loader = DataLoader(
            start_date=self.config['data']['start_date'],
            end_date=self.config['data']['end_date'],
            symbols=self.config['data']['symbols'],
            data_dir='data',
            lookback_days=120
        )
        
        self.lim_calculator = LIMCalculator(
            window_size=self.config['lim']['window_size'],
            alpha=self.config['lim']['alpha'],
            lookback_days=self.config['lim']['lookback_days']
        )
        
        self.max_days = 100
        
    def calculate_lim_data(self):
        """Calculate LIM data."""
        print("Loading data and calculating LIM time series...")
        
        self.data_loader.load_data()
        all_stock_data = self.data_loader.all_stock_data
        
        start_date = pd.to_datetime(self.config['data']['start_date'])
        end_date = pd.to_datetime(self.config['data']['end_date'])
        
        lim_results = {}
        stock_names = {}
        
        print(f"Calculating LIM for {len(self.config['data']['symbols'])} stocks...")
        
        for i, symbol in enumerate(self.config['data']['symbols'], 1):
            if symbol in all_stock_data:
                print(f"  [{i:2d}/30] Processing {symbol}...")
                
                stock_data = all_stock_data[symbol]
                
                if 'date' not in stock_data.columns:
                    continue
                
                stock_data = stock_data.sort_values('date').reset_index(drop=True)
                stock_data['date'] = pd.to_datetime(stock_data['date'])
                
                target_mask = (stock_data['date'] >= start_date) & (stock_data['date'] <= end_date)
                target_indices = stock_data[target_mask].index.tolist()
                
                if len(target_indices) < self.max_days:
                    print(f"    Warning: Insufficient data for {symbol}")
                    continue
                
                daily_lim_values = []
                calculation_indices = target_indices[:self.max_days]
                
                for j, target_idx in enumerate(calculation_indices, 1):
                    current_date = stock_data.iloc[target_idx]['date']
                    historical_data = stock_data.iloc[:target_idx + 1].copy()
                    
                    try:
                        lim_value = self.lim_calculator.calculate_LIM_star(
                            historical_data, 
                            current_date, 
                            lookback_days=self.lim_calculator.lookback_days
                        )
                        
                        if np.isfinite(lim_value) and lim_value > 0:
                            daily_lim_values.append(lim_value)
                        else:
                            daily_lim_values.append(np.nan)
                        
                    except Exception:
                        daily_lim_values.append(np.nan)
                
                lim_results[symbol] = daily_lim_values
                stock_names[symbol] = symbol
                print(f"    Completed: {len(daily_lim_values)} LIM values calculated")
        
        if lim_results:
            max_length = max(len(values) for values in lim_results.values())
            
            for symbol in lim_results:
                while len(lim_results[symbol]) < max_length:
                    lim_results[symbol].append(np.nan)
                lim_results[symbol] = lim_results[symbol][:max_length]
            
            self.lim_df = pd.DataFrame(lim_results, index=range(1, max_length + 1))
            self.lim_df.index.name = 'Trading Day'
            self.stock_names = stock_names
            
            print(f"Successfully created LIM DataFrame: {len(self.lim_df.columns)} stocks × {len(self.lim_df)} days")
        else:
            raise ValueError("No valid LIM data calculated")
    
    def create_lim_curves_chart(self):
        """Create LIM curves chart."""
        print("Creating LIM curves visualization...")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
        
        avg_lim = self.lim_df.mean(axis=1, skipna=True)
        median_lim = self.lim_df.median(axis=1, skipna=True)
        std_lim = self.lim_df.std(axis=1, skipna=True)
        
        ax1.set_title('LIM Time Series for All 30 Semiconductor Stocks\n' +
                     f'Individual Stock Trajectories (90-Day Lookback Period)', 
                     fontsize=14, fontweight='bold', pad=15)
        
        colors = plt.cm.tab20(np.linspace(0, 1, len(self.lim_df.columns)))
        
        for i, (symbol, color) in enumerate(zip(self.lim_df.columns, colors)):
            data = self.lim_df[symbol].dropna()
            if len(data) > 0:
                ax1.plot(data.index, data.values, 
                        color=color, alpha=0.6, linewidth=1.2, 
                        label=symbol if i < 10 else None)
        
        ax1.plot(avg_lim.index, avg_lim.values, 
                color='red', linewidth=3, alpha=0.9, 
                label='Portfolio Average', zorder=10)
        
        ax1.set_xlabel('Trading Day', fontsize=12, fontweight='bold')
        ax1.set_ylabel('LIM Value', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=9, loc='upper right', ncol=2)
        
        ax2.set_title('LIM Statistical Analysis: Central Tendency and Dispersion\n' +
                     'Portfolio-Level Aggregation and Variability', 
                     fontsize=14, fontweight='bold', pad=15)
        
        ax2.plot(avg_lim.index, avg_lim.values, 
                color='red', linewidth=3, label='Mean LIM', alpha=0.9)
        ax2.plot(median_lim.index, median_lim.values, 
                color='blue', linewidth=2, label='Median LIM', alpha=0.8)
        
        upper_band = avg_lim + std_lim
        lower_band = avg_lim - std_lim
        
        ax2.fill_between(avg_lim.index, lower_band, upper_band, 
                        color='red', alpha=0.2, label='±1 Std Dev')
        
        if 30 in avg_lim.index:
            ax2.axvline(x=30, color='orange', linestyle='--', alpha=0.8, linewidth=2)
            ax2.text(30, ax2.get_ylim()[1]*0.9, '30 Days\n(Monthly Cycle)', 
                    ha='center', fontweight='bold', color='orange',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        if 60 in avg_lim.index:
            ax2.axvline(x=60, color='green', linestyle='--', alpha=0.8, linewidth=2)
            ax2.text(60, ax2.get_ylim()[1]*0.8, '60 Days\n(Bi-Monthly)', 
                    ha='center', fontweight='bold', color='green',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        ax2.set_xlabel('Trading Day', fontsize=12, fontweight='bold')
        ax2.set_ylabel('LIM Value', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=11, loc='upper right')
        
        stats_text = f"""LIM Statistics Summary:
• Stocks Analyzed: {len(self.lim_df.columns)}
• Time Period: {len(self.lim_df)} trading days
• Mean LIM Range: {avg_lim.min():.3f} - {avg_lim.max():.3f}
• Average Volatility: {std_lim.mean():.3f}
• Data Completeness: {(1 - self.lim_df.isnull().sum().sum() / (len(self.lim_df) * len(self.lim_df.columns))):.1%}

Key Observations:
• Peak LIM periods indicate optimal rebalancing windows
• Volatility patterns reveal market cycle characteristics
• Cross-stock correlations suggest systematic factors"""
        
        ax2.text(0.02, 0.02, stats_text, transform=ax2.transAxes, fontsize=10,
                verticalalignment='bottom', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.9))
        
        plt.tight_layout()
        
        output_dir = 'visualization'
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f'lim_curves_30stocks_{timestamp}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"LIM curves chart saved to: {output_path}")
        
        print("\n" + "="*60)
        print("LIM DATA STATISTICS SUMMARY")
        print("="*60)
        print(f"Number of stocks: {len(self.lim_df.columns)}")
        print(f"Time series length: {len(self.lim_df)} trading days")
        print(f"Total data points: {len(self.lim_df) * len(self.lim_df.columns)}")
        print(f"Missing values: {self.lim_df.isnull().sum().sum()}")
        print(f"Data completeness: {(1 - self.lim_df.isnull().sum().sum() / (len(self.lim_df) * len(self.lim_df.columns))):.1%}")
        print()
        print("Portfolio Average LIM Statistics:")
        print(f"  Mean: {avg_lim.mean():.4f}")
        print(f"  Std:  {avg_lim.std():.4f}")
        print(f"  Min:  {avg_lim.min():.4f}")
        print(f"  Max:  {avg_lim.max():.4f}")
        print(f"  Range: {avg_lim.max() - avg_lim.min():.4f}")
        print("="*60)
        
        plt.show()
        
        return output_path, self.lim_df
    
    def calculate_lim_changes(self):
        """Calculate LIM changes (LIMt - LIMt-1)."""
        print("Calculating LIM changes...")
        
        lim_changes = self.lim_df.diff()
        
        lim_changes = lim_changes.iloc[1:]
        
        avg_changes = lim_changes.mean(axis=1, skipna=True)
        median_changes = lim_changes.median(axis=1, skipna=True)
        std_changes = lim_changes.std(axis=1, skipna=True)
        
        return lim_changes, avg_changes, median_changes, std_changes

    def create_lim_changes_chart(self):
        """Create LIM changes chart."""
        print("Creating LIM changes visualization...")
        
        lim_changes, avg_changes, median_changes, std_changes = self.calculate_lim_changes()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
        
        ax1.set_title('Daily LIM Changes for All 30 Semiconductor Stocks\n' +
                     f'Individual Stock Changes (90-Day Lookback Period)', 
                     fontsize=14, fontweight='bold', pad=15)
        
        colors = plt.cm.tab20(np.linspace(0, 1, len(lim_changes.columns)))
        
        for i, (symbol, color) in enumerate(zip(lim_changes.columns, colors)):
            data = lim_changes[symbol].dropna()
            if len(data) > 0:
                ax1.plot(data.index, data.values, 
                        color=color, alpha=0.6, linewidth=1.2, 
                        label=symbol if i < 10 else None)
        
        ax1.plot(avg_changes.index, avg_changes.values, 
                color='red', linewidth=3, alpha=0.9, 
                label='Portfolio Average Change', zorder=10)
        
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        
        ax1.set_xlabel('Trading Day', fontsize=12, fontweight='bold')
        ax1.set_ylabel('LIM Change', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=9, loc='upper right', ncol=2)
        
        ax2.set_title('LIM Changes Statistical Analysis\n' +
                     'Portfolio-Level Aggregation and Variability', 
                     fontsize=14, fontweight='bold', pad=15)
        
        ax2.plot(avg_changes.index, avg_changes.values, 
                color='red', linewidth=3, label='Mean Change', alpha=0.9)
        ax2.plot(median_changes.index, median_changes.values, 
                color='blue', linewidth=2, label='Median Change', alpha=0.8)
        
        upper_band = avg_changes + std_changes
        lower_band = avg_changes - std_changes
        
        ax2.fill_between(avg_changes.index, lower_band, upper_band, 
                        color='red', alpha=0.2, label='±1 Std Dev')
        
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        
        if 30 in avg_changes.index:
            ax2.axvline(x=30, color='orange', linestyle='--', alpha=0.8, linewidth=2)
            ax2.text(30, ax2.get_ylim()[1]*0.9, '30 Days\n(Monthly Cycle)', 
                    ha='center', fontweight='bold', color='orange',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        if 60 in avg_changes.index:
            ax2.axvline(x=60, color='green', linestyle='--', alpha=0.8, linewidth=2)
            ax2.text(60, ax2.get_ylim()[1]*0.8, '60 Days\n(Bi-Monthly)', 
                    ha='center', fontweight='bold', color='green',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        ax2.set_xlabel('Trading Day', fontsize=12, fontweight='bold')
        ax2.set_ylabel('LIM Change', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=11, loc='upper right')
        
        stats_text = f"""LIM Changes Statistics Summary:
• Stocks Analyzed: {len(lim_changes.columns)}
• Time Period: {len(lim_changes)} trading days
• Mean Change Range: {avg_changes.min():.3f} - {avg_changes.max():.3f}
• Average Volatility: {std_changes.mean():.3f}
• Data Completeness: {(1 - lim_changes.isnull().sum().sum() / (len(lim_changes) * len(lim_changes.columns))):.1%}

Key Observations:
• Positive changes indicate increasing LIM values
• Negative changes suggest decreasing LIM values
• Volatility patterns reveal market sensitivity"""
        
        ax2.text(0.02, 0.02, stats_text, transform=ax2.transAxes, fontsize=10,
                verticalalignment='bottom', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.9))
        
        plt.tight_layout()
        
        output_dir = 'visualization'
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f'lim_changes_30stocks_{timestamp}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"LIM changes chart saved to: {output_path}")
        
        print("\n" + "="*60)
        print("LIM CHANGES STATISTICS SUMMARY")
        print("="*60)
        print(f"Number of stocks: {len(lim_changes.columns)}")
        print(f"Time series length: {len(lim_changes)} trading days")
        print(f"Total data points: {len(lim_changes) * len(lim_changes.columns)}")
        print(f"Missing values: {lim_changes.isnull().sum().sum()}")
        print(f"Data completeness: {(1 - lim_changes.isnull().sum().sum() / (len(lim_changes) * len(lim_changes.columns))):.1%}")
        print()
        print("Portfolio Average Change Statistics:")
        print(f"  Mean: {avg_changes.mean():.4f}")
        print(f"  Std:  {avg_changes.std():.4f}")
        print(f"  Min:  {avg_changes.min():.4f}")
        print(f"  Max:  {avg_changes.max():.4f}")
        print(f"  Range: {avg_changes.max() - avg_changes.min():.4f}")
        print("="*60)
        
        plt.show()
        
        return output_path, lim_changes
    
    def analyze_lim_periodicity(self):
        """Analyze LIM periodicity."""
        print("Analyzing LIM periodicity...")
        
        portfolio_avg = self.lim_df.mean(axis=1, skipna=True)
        
        portfolio_data = portfolio_avg.dropna().values
        
        from statsmodels.tsa.stattools import acf
        from scipy.signal import find_peaks
        acf_values = acf(portfolio_data, nlags=60)
        
        from scipy.fft import fft
        fft_values = fft(portfolio_data)
        freqs = np.fft.fftfreq(len(portfolio_data))
        
        rolling_corr = portfolio_avg.rolling(window=30).corr(portfolio_avg.shift(30))
        
        peaks, _ = find_peaks(portfolio_data, distance=10)
        valleys, _ = find_peaks(-portfolio_data, distance=10)
        
        cycle_length = 30
        cycle_means = []
        cycle_stds = []
        for i in range(0, len(portfolio_avg), cycle_length):
            cycle_data = portfolio_avg.iloc[i:i+cycle_length]
            if len(cycle_data) >= 20:
                cycle_means.append(cycle_data.mean())
                cycle_stds.append(cycle_data.std())
        
        rebalance_points = list(range(30, len(portfolio_avg), 30))
        rebalance_values = portfolio_avg.iloc[rebalance_points] if rebalance_points else []
        
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(20, 18))
        
        ax1.set_title('LIM Autocorrelation Analysis\n' +
                     'Identifying Periodic Patterns', 
                     fontsize=12, fontweight='bold', pad=10)
        ax1.plot(range(len(acf_values)), acf_values, 'b-', linewidth=2)
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax1.axvline(x=30, color='red', linestyle='--', alpha=0.8, linewidth=2)
        ax1.text(30, max(acf_values)*0.8, '30 Days', ha='center', fontweight='bold', color='red')
        ax1.set_xlabel('Lag (Days)', fontsize=10)
        ax1.set_ylabel('Autocorrelation', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        ax2.set_title('LIM Frequency Analysis\n' +
                     'Dominant Periodic Components', 
                     fontsize=12, fontweight='bold', pad=10)
        positive_freq_mask = freqs > 0
        periods = 1 / freqs[positive_freq_mask]
        valid_mask = (periods >= 5) & (periods <= 100)
        ax2.plot(periods[valid_mask], np.abs(fft_values[positive_freq_mask])[valid_mask], 'g-', linewidth=2)
        ax2.axvline(x=30, color='red', linestyle='--', alpha=0.8, linewidth=2)
        ax2.text(30, ax2.get_ylim()[1]*0.8, '30 Days', ha='center', fontweight='bold', color='red')
        ax2.set_xlabel('Period (Days)', fontsize=10)
        ax2.set_ylabel('Amplitude', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        ax3.set_title('LIM Peaks and Valleys Analysis\n' +
                     'Market Turning Points', 
                     fontsize=12, fontweight='bold', pad=10)
        ax3.plot(portfolio_avg.index, portfolio_avg.values, 'b-', linewidth=1, alpha=0.7)
        if len(peaks) > 0:
            ax3.plot(portfolio_avg.index[peaks], portfolio_avg.iloc[peaks], 'ro', markersize=6, label=f'Peaks ({len(peaks)})')
        if len(valleys) > 0:
            ax3.plot(portfolio_avg.index[valleys], portfolio_avg.iloc[valleys], 'go', markersize=6, label=f'Valleys ({len(valleys)})')
        
        for point in rebalance_points:
            if point < len(portfolio_avg):
                ax3.axvline(x=point, color='orange', linestyle=':', alpha=0.6)
        
        ax3.set_xlabel('Trading Day', fontsize=10)
        ax3.set_ylabel('LIM Value', fontsize=10)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        ax4.set_title('30-Day Rolling Correlation\n' +
                     'Periodic Pattern Strength', 
                     fontsize=12, fontweight='bold', pad=10)
        ax4.plot(rolling_corr.index, rolling_corr.values, 'r-', linewidth=2)
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax4.axhline(y=rolling_corr.mean(), color='green', linestyle='--', alpha=0.8, 
                   label=f'Mean: {rolling_corr.mean():.3f}')
        ax4.set_xlabel('Trading Day', fontsize=10)
        ax4.set_ylabel('Correlation', fontsize=10)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        ax5.set_title('30-Day Cycle Aggregation\n' +
                     'Rebalancing Period Analysis', 
                     fontsize=12, fontweight='bold', pad=10)
        cycle_numbers = range(1, len(cycle_means) + 1)
        ax5.plot(cycle_numbers, cycle_means, 'bo-', linewidth=2, markersize=6, label='Cycle Mean')
        ax5.fill_between(cycle_numbers, 
                        np.array(cycle_means) - np.array(cycle_stds),
                        np.array(cycle_means) + np.array(cycle_stds),
                        alpha=0.3, label='±1 Std Dev')
        ax5.set_xlabel('30-Day Cycle Number', fontsize=10)
        ax5.set_ylabel('Average LIM Value', fontsize=10)
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        ax6.set_title('Rebalancing Points Analysis\n' +
                     'LIM Values at 30-Day Intervals', 
                     fontsize=12, fontweight='bold', pad=10)
        if len(rebalance_values) > 0:
            ax6.plot(range(len(rebalance_values)), rebalance_values, 'ro-', 
                    linewidth=2, markersize=8, label='Rebalancing Points')
            ax6.axhline(y=rebalance_values.mean(), color='green', linestyle='--', 
                       alpha=0.8, label=f'Mean: {rebalance_values.mean():.3f}')
            
            cv = rebalance_values.std() / rebalance_values.mean()
            ax6.text(0.02, 0.98, f'Coefficient of Variation: {cv:.3f}', 
                    transform=ax6.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.9))
        
        ax6.set_xlabel('Rebalancing Event', fontsize=10)
        ax6.set_ylabel('LIM Value', fontsize=10)
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_dir = 'visualization'
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f'lim_periodicity_analysis_{timestamp}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Periodicity analysis chart saved to: {output_path}")
        
        peak_intervals = np.diff(peaks) if len(peaks) > 1 else []
        valley_intervals = np.diff(valleys) if len(valleys) > 1 else []
        
        print("\n" + "="*80)
        print("LIM PERIODICITY ANALYSIS SUMMARY")
        print("="*80)
        print("Basic statistics:")
        print(f"  • 30-day autocorrelation: {acf_values[30]:.4f}")
        print(f"  • Mean 30-day rolling correlation: {rolling_corr.mean():.4f}")
        if len(rebalance_values) > 0:
            print(f"  • Mean LIM at rebalance points: {rebalance_values.mean():.4f}")
            print(f"  • Coefficient of variation at rebalance points: {rebalance_values.std()/rebalance_values.mean():.4f}")
        
        print("\nPeaks and valleys:")
        print(f"  • Peaks detected: {len(peaks)}")
        print(f"  • Valleys detected: {len(valleys)}")
        if len(peak_intervals) > 0:
            print(f"  • Mean peak interval: {np.mean(peak_intervals):.1f} days")
        if len(valley_intervals) > 0:
            print(f"  • Mean valley interval: {np.mean(valley_intervals):.1f} days")
        
        print("\n30-day cycle analysis:")
        print(f"  • Complete 30-day cycles: {len(cycle_means)}")
        if len(cycle_means) > 0:
            print(f"  • Mean LIM within cycle: {np.mean(cycle_means):.4f}")
            print(f"  • LIM std within cycle: {np.std(cycle_means):.4f}")
        
        print("\nEvidence supporting 30-day rebalancing:")
        if acf_values[30] > 0.1:
            print(f"  ✓ 30-day autocorrelation is significant ({acf_values[30]:.3f})")
        else:
            print(f"  ✗ 30-day autocorrelation is weak ({acf_values[30]:.3f})")
        
        if rolling_corr.mean() > 0.05:
            print(f"  ✓ 30-day rolling correlation is stable ({rolling_corr.mean():.3f})")
        else:
            print(f"  ✗ 30-day rolling correlation is unstable ({rolling_corr.mean():.3f})")
        
        if len(cycle_means) > 2 and np.std(cycle_means) / np.mean(cycle_means) < 0.5:
            print("  ✓ LIM values are relatively stable within 30-day cycles")
        else:
            print("  ✗ LIM values vary substantially within 30-day cycles")
        
        print("="*80)
        
        plt.show()
        
        return output_path, {
            'acf': acf_values,
            'fft': fft_values,
            'freqs': freqs,
            'rolling_corr': rolling_corr,
            'peaks': peaks,
            'valleys': valleys,
            'cycle_means': cycle_means,
            'cycle_stds': cycle_stds,
            'rebalance_values': rebalance_values
        }
    
    def run(self):
        """Run the full workflow."""
        print("=" * 60)
        print("30-STOCK LIM VALUES AND PERIODICITY ANALYSIS")
        print("=" * 60)
        
        self.calculate_lim_data()
        
        chart_path, lim_data = self.create_lim_curves_chart()
        
        periodicity_path, periodicity_data = self.analyze_lim_periodicity()
        
        print("=" * 60)
        print(f"Visualization complete!")
        print(f"LIM curves chart saved to: {chart_path}")
        print(f"Periodicity analysis chart saved to: {periodicity_path}")
        print("=" * 60)
        
        return chart_path, lim_data, periodicity_path, periodicity_data


def main():
    """Main entry."""
    try:
        visualizer = LIMCurvesVisualizer()
        chart_path, lim_data, periodicity_path, periodicity_data = visualizer.run()
        return chart_path, lim_data, periodicity_path, periodicity_data
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None


if __name__ == "__main__":
    main() 

