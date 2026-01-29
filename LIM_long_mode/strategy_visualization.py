import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib as mpl
from matplotlib.dates import DateFormatter

def plot_strategy_comparison(csv_path, output_dir, dpi=600):
    """Plot strategy comparison chart."""
    # Set global font to Times New Roman
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False
    
    # Load data
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded data with {len(df)} rows")
    except Exception as e:
        print(f"Error reading CSV: {str(e)}")
        return
    
    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Extract strategy names (all columns except 'date')
    strategies = [col for col in df.columns if col != 'date']
    
    # Create figure
    plt.figure(figsize=(5, 4), facecolor='white')
    
    # Color palette - 14 distinct colors
    colors = [
        '#1f77b4',  # Blue
        '#2ca02c',  # Green
        '#ff7f0e',  # Orange
        '#9467bd',  # Purple
        '#8c564b',  # Brown
        '#e377c2',  # Pink
        '#7f7f7f',  # Gray
        '#bcbd22',  # Yellow-green
        '#17becf',  # Cyan
        '#aec7e8',  # Light blue
        '#ffbb78',  # Light orange
        '#98df8a',  # Light green
        '#ff9896',  # Light red
        '#c5b0d5'   # Light purple
    ]
    
    # Plot non-baseline strategies first
    for i, strategy in enumerate(strategies):
        if strategy.lower() != 'baseline':
            try:
                values = df[strategy].astype(float)
                # Normalize
                if len(values) > 0 and values.iloc[0] != 0:
                    values = values / values.iloc[0]
                    plt.plot(df['date'], values, label=strategy, 
                            linewidth=1.0, color=colors[i % len(colors)])
            except Exception as e:
                print(f"Error processing strategy {strategy}: {str(e)}")
    
    # Plot baseline strategy last
    if 'baseline' in [s.lower() for s in strategies]:
        baseline_values = df['baseline'].astype(float)
        if len(baseline_values) > 0 and baseline_values.iloc[0] != 0:
            baseline_values = baseline_values / baseline_values.iloc[0]
            plt.plot(df['date'], baseline_values, label='Proposed DASRL', 
                    color='red', linewidth=1.5)
    
    # Axis settings
    ax = plt.gca()
    
    # Set date ticks
    date_min = df['date'].min()
    date_max = df['date'].max()
    date_range = (date_max - date_min).days
    interval = date_range // 4
    date_ticks = [date_min + pd.Timedelta(days=i*interval) for i in range(5)]
    if date_ticks[-1] > date_max:
        date_ticks[-1] = date_max
    
    ax.set_xticks(date_ticks)
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
    plt.xticks(rotation=45, ha='right')
    
    # Add border
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.title('Strategy Comparison', fontsize=10, fontweight='bold', pad=10)
    plt.xlabel('Date', fontsize=10)
    plt.ylabel('Portfolio Value', fontsize=10)
    plt.legend(frameon=False, fontsize=6, loc='upper left')
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, 'portfolio_comparison.jpg'), 
                dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"Strategy comparison plot saved to: {output_dir}")

if __name__ == '__main__':
    # Set paths
    csv_path = r"D:\RL experiment\LIMPPO_CNN\0419_main_experiment_results\strategy comparison.csv"
    output_dir = r"D:\RL experiment\LIMPPO_CNN\0419_main_experiment_results"
    
    # Plot strategy comparison
    plot_strategy_comparison(csv_path, output_dir, dpi=600) 

