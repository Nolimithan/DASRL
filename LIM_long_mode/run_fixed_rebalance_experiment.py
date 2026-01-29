"""Fixed rebalance interval experiment - TR curve visualization."""

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
import matplotlib.pyplot as plt
from scipy import stats
import warnings
import subprocess
import re
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description='Fixed rebalance TR curve visualization')
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'],
                      help='Run mode: train or test (default: test)')
    parser.add_argument('--config', type=str, default='config/default.yaml',
                      help='Config file path')
    parser.add_argument('--start_date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--group_type', type=str, default='TOP',
                      choices=['TOP', 'MIDDLE', 'LOW'],
                      help='Stock group type')
    parser.add_argument('--runs', type=int, default=10,
                      help='Runs per strategy (default: 10)')
    parser.add_argument('--seeds', type=int, nargs='+',
                      default=[888, 1161, 4242, 4320, 5437, 5497, 5992, 8510, 9859, 9941],
                      help='Random seed list (paper defaults)')
    return parser.parse_args()


RANDOM_SEEDS = [888, 1161, 4242, 4320, 5437, 5497, 5992, 8510, 9859, 9941]

MULTISEED_RUNS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                   "multiseed_runs", "long_cn_semiconductor", "group_TOP")


def load_30day_results_from_multiseed():
    print("Loading 30-day rebalance results from multiseed_runs...")
    print(f"  Directory: {MULTISEED_RUNS_DIR}")
    
    returns = []
    loaded_seeds = []
    
    for seed in RANDOM_SEEDS:
        possible_paths = [
            os.path.join(MULTISEED_RUNS_DIR, f"seed_{seed}", "performance_metrics.csv"),
            os.path.join(MULTISEED_RUNS_DIR, f"seed_{seed}", "results", "test_results_*", "performance_metrics.csv"),
        ]
        
        metrics_file = None
        for path_pattern in possible_paths:
            if '*' in path_pattern:
                import glob
                matches = glob.glob(path_pattern)
                if matches:
                    matches.sort(key=os.path.getmtime, reverse=True)
                    metrics_file = matches[0]
                    break
            elif os.path.exists(path_pattern):
                metrics_file = path_pattern
                break
        
        if metrics_file and os.path.exists(metrics_file):
            try:
                df = pd.read_csv(metrics_file)
                if 'total_return' in df.columns:
                    tr = df['total_return'].iloc[0]
                    returns.append(tr)
                    loaded_seeds.append(seed)
                    print(f"  ✓ seed_{seed}: TR = {tr:.4f}")
                else:
                    print(f"  ✗ seed_{seed}: performance_metrics.csv missing total_return")
            except Exception as e:
                print(f"  ✗ seed_{seed}: read failed - {e}")
        else:
            print(f"  ✗ seed_{seed}: performance_metrics.csv not found")
    
    print(f"  Loaded {len(returns)}/{len(RANDOM_SEEDS)} seeds")
    return returns, loaded_seeds


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def run_single_strategy(config, data_loader, lim_calculator, rebalance_period, run_id=0, seed=None):
    if seed is None:
        seed = RANDOM_SEEDS[run_id % len(RANDOM_SEEDS)]
    
    print(f"Running {rebalance_period}-day rebalance (run {run_id+1}, seed={seed})...")
    np.random.seed(seed)
    
    print("  → Using fixed rebalance config")
    strategy_config = config.copy()
    strategy_config['environment']['rebalance_period'] = rebalance_period
    strategy_config['environment']['dynamic_rebalance'] = False
    
    if 'random_rebalance' in strategy_config['environment']:
        strategy_config['environment']['random_rebalance'] = False
    env = TradingEnvironment(
        config=strategy_config,
        data_loader=data_loader,
        lim_calculator=lim_calculator
    )
    
    model = PPOModel(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        lr=strategy_config['rl']['learning_rate'],
        gamma=strategy_config['rl']['gamma'],
        clip_ratio=strategy_config['rl']['clip_ratio'],
        value_coef=strategy_config['rl']['value_coef'],
        entropy_coef=strategy_config['rl']['entropy_coef']
    )
    
    
    if config.get('mode', 'test') == 'train':
        model.train(env, epochs=strategy_config['rl']['epochs'])
    else:
        model_path = config['paths']['model_load_path']
        model.load(model_path)
        print(f"  ○ Loaded pretrained model: {model_path}")
    
    
    from utils.portfolio import Portfolio
    portfolio = Portfolio(
        initial_capital=strategy_config['portfolio']['initial_capital'],
        commission_rate=strategy_config['trading']['commission_rate']
    )
    
    
    env.reset()
    portfolio.run_strategy(model, env)
    
    
    results = portfolio.get_performance_summary()
    total_return = results.get('total_return', 0)
    
    print(f"  Done: TR = {total_return:.4f} ({total_return*100:.2f}%)")
    
    return total_return

def run_baseline_verification(config, data_loader, lim_calculator, quiet=False):
    if not quiet:
        print("Verifying main baseline (30-day rebalance)...")
    
    
    env = TradingEnvironment(
        config=config,
        data_loader=data_loader,
        lim_calculator=lim_calculator
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
    
    
    if config.get('mode', 'test') == 'train':
        model.train(env, epochs=config['rl']['epochs'])
    else:
        model.load(config['paths']['model_load_path'])
        if not quiet:
            print(f"  Loaded main pretrained model: {config['paths']['model_load_path']}")
    
    
    from utils.portfolio import Portfolio
    portfolio = Portfolio(
        initial_capital=config['portfolio']['initial_capital'],
        commission_rate=config['trading']['commission_rate']
    )
    
    
    env.reset()
    portfolio.run_strategy(model, env)
    
    
    results = portfolio.get_performance_summary()
    total_return = results.get('total_return', 0)
    
    if not quiet:
        print(f"  Main baseline TR = {total_return:.4f} ({total_return*100:.2f}%)")
        print(f"  Config: rebalance_period={config['environment']['rebalance_period']}")
    
    return total_return

def plot_tr_curves(tr_data, output_dir):
    print("Generating TR curve visualization...")
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Times New Roman'
    plt.rcParams['font.size'] = 16
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['xtick.major.width'] = 1.5
    plt.rcParams['ytick.major.width'] = 1.5 
    periods = sorted(tr_data.keys())
    means = []
    stds = []
    
    for period in periods:
        returns = tr_data[period]
        means.append(np.mean(returns))
        stds.append(np.std(returns))
    
    means = np.array(means)
    stds = np.array(stds)
    
    confidence_level = 0.95
    alpha = 1 - confidence_level
    n = len(tr_data[periods[0]])
    df_freedom = n - 1
    t_critical = stats.t.ppf(1 - alpha/2, df_freedom)
    
    se = stds / np.sqrt(n)
    ci_lower = means - t_critical * se
    ci_upper = means + t_critical * se
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    ax.plot(periods, means, 'b-', linewidth=3, marker='o', markersize=8, 
            label='Mean Total Return', color='#1f77b4')
    
    ax.fill_between(periods, ci_lower, ci_upper, alpha=0.2, color='lightblue', 
                    label=f'{confidence_level*100:.0f}% Confidence Interval')
    
    ax.set_xlabel('Rebalancing period (days)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Total return', fontsize=14, fontweight='bold')
    ax.set_title('Rebalancing period analysis', 
                 fontsize=16, fontweight='bold', pad=20)
    
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    ax.grid(True, alpha=0.3, linestyle='--')
    
    ax.legend(loc='best', fontsize=11, frameon=True, fancybox=True, shadow=True)
    
    ax.set_xlim([min(periods)-2, max(periods)+2])
    y_range = max(means) - min(means)
    ax.set_ylim([min(means) - 0.1*y_range, max(means) + 0.1*y_range])
    
    plt.tight_layout()
    tr_curve_path = os.path.join(output_dir, 'TR_curves_analysis.jpg')
    plt.savefig(tr_curve_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"TR curve saved to: {tr_curve_path}")
    
    report_path = os.path.join(output_dir, 'TR_analysis_summary.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("Total Return Analysis Summary\n")
        f.write("="*50 + "\n\n")
        
        f.write(f"Experiment Overview:\n")
        f.write(f"- Periods tested: {min(periods)} to {max(periods)} days\n")
        f.write(f"- Sample size: {n} runs per period\n")
        f.write(f"- Main experiment: 30-day rebalancing\n\n")
        
        f.write("Key Findings:\n")
        best_period = periods[np.argmax(means)]
        worst_period = periods[np.argmin(means)]
        f.write(f"- Best performing period: {best_period} days (TR = {max(means):.4f})\n")
        f.write(f"- Worst performing period: {worst_period} days (TR = {min(means):.4f})\n")
        
        if 30 in tr_data:
            idx_30 = periods.index(30)
            rank = sorted(enumerate(means), key=lambda x: x[1], reverse=True)
            rank_30 = next(i for i, (idx, _) in enumerate(rank) if idx == idx_30) + 1
            f.write(f"- 30-day strategy rank: {rank_30}/{len(periods)} (TR = {means[idx_30]:.4f})\n")
        
        all_returns = []
        for returns in tr_data.values():
            all_returns.extend(returns)
        
        skewness = stats.skew(all_returns)
        kurtosis = stats.kurtosis(all_returns)
        
        f.write(f"\nDistribution Analysis (All Data):\n")
        f.write(f"- Skewness: {skewness:.3f}")
        if abs(skewness) > 0.5:
            f.write(" (Significantly skewed)")
        else:
            f.write(" (Approximately symmetric)")
        f.write(f"\n- Kurtosis: {kurtosis:.3f}")
        if kurtosis > 3:
            f.write(" (Leptokurtic - Fat tails)")
        else:
            f.write(" (Platykurtic - Thin tails)")
        
        f.write(f"\n\nConclusion:\n")
        if abs(skewness) > 0.5 or kurtosis > 3:
            f.write("The total return distribution exhibits fat-tail characteristics.\n")
        else:
            f.write("The total return distribution is approximately normal.\n")
    
    print(f"Statistics summary saved to: {report_path}")

def run_main_experiment_command(args, run_id=0):
    print(f"  Running main experiment command (run {run_id+1})...")
    
    
    cmd = [
        "python", "main.py",
        "--mode", "test",
        "--start_date", args.start_date if args.start_date else "2024-04-06",
        "--end_date", args.end_date if args.end_date else "2025-04-06", 
        "--config", args.config,
        "--visualize",
        "--group_type", args.group_type
    ]
    
    print(f"    Command: {' '.join(cmd)}")
    
    try:
        
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=300,
            cwd=os.getcwd()
        )
        
        if result.returncode != 0:
            print(f"    Command failed, return code: {result.returncode}")
            print(f"    Error: {result.stderr}")
            return None
        
        output = result.stdout
        
        
        tr_patterns = [
            r"Total Return[:\s]*([0-9.-]+)",
            r"Total Return[:\s]*([0-9.-]+)", 
            r"TR[:\s]*([0-9.-]+)",
            r"total_return[:\s]*([0-9.-]+)"
        ]
        
        tr_value = None
        for pattern in tr_patterns:
            matches = re.findall(pattern, output, re.IGNORECASE)
            if matches:
                try:
                    tr_value = float(matches[-1])
                    break
                except ValueError:
                    continue
        
        if tr_value is not None:
            print(f"    Done: TR = {tr_value:.4f}")
            return tr_value
        else:
            print("    Warning: failed to extract TR from output")
            print("    Output:")
            print(output[-500:])
            return None
            
    except subprocess.TimeoutExpired:
        print("    Command timed out (5 minutes)")
        return None
    except Exception as e:
        print(f"    Command error: {e}")
        return None

def main():
    global RANDOM_SEEDS
    args = parse_args()
    
    RANDOM_SEEDS = args.seeds
    
    print("=" * 60)
    print("Fixed rebalance TR curve analysis")
    print("=" * 60)
    print(f"Run mode: {args.mode}")
    print(f"Stock group: {args.group_type}")
    print(f"Runs per strategy: {args.runs}")
    print(f"Random seeds: {RANDOM_SEEDS}")
    print()
    print("Experiment design:")
    print("- 30-day rebalance: load 10 precomputed seeds from multiseed_runs")
    print("- Other periods: use fixed rebalance config and pretrained model")
    print("- Purpose: analyze rebalance frequency impact")
    print()
    
    config = load_config(args.config)
    config['mode'] = args.mode
    
    if args.start_date:
        config['data']['start_date'] = args.start_date
    if args.end_date:
        config['data']['end_date'] = args.end_date
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config['paths']['results_path'], f'TR_analysis_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    print("Initializing data loader...")
    data_dir = os.path.join(os.getcwd(), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    data_loader = DataLoader(
        start_date=config['data']['start_date'],
        end_date=config['data']['end_date'],
        symbols=config['data']['symbols'],
        data_dir=data_dir
    )
    data_loader.load_data()
    
    lim_calculator = LIMCalculator(
        window_size=config['lim']['window_size'],
        alpha=config['lim']['alpha'],
        lookback_days=config['lim']['lookback_days']
    )
    
    print("\n" + "=" * 60)
    print("Running fixed rebalance experiments")
    print("=" * 60)
    
    rebalance_periods = list(range(10, 95, 5))
    if 30 not in rebalance_periods:
        rebalance_periods.append(30)
        rebalance_periods.sort()
    
    print(f"Tested rebalance periods: {rebalance_periods}")
    print(f"Total periods: {len(rebalance_periods)}")
    print("Note: 30-day period loads precomputed results; others run fixed rebalance")
    print("All periods run multiple trials to estimate confidence intervals")
    print()
    
    if 30 in rebalance_periods:
        index_30 = rebalance_periods.index(30)
        print(f"✓ 30-day period in range (index {index_30+1}), runs={args.runs}")
    else:
        print("❌ 30-day period not in test range!")
        return
    print()
    
    tr_data = {}
    
    for period in rebalance_periods:
        print(f"Testing {period}-day rebalance period...")
        if period == 30:
            print("  → Main period; loading precomputed multiseed results")
            period_returns, loaded_seeds = load_30day_results_from_multiseed()
            
            if len(period_returns) == 0:
                print("  ⚠️ Warning: no 30-day results loaded from multiseed_runs")
                print("  → Falling back to running experiments...")
                period_returns = []
                for run_id in range(args.runs):
                    try:
                        seed = RANDOM_SEEDS[run_id % len(RANDOM_SEEDS)]
                        total_return = run_baseline_verification(config, data_loader, lim_calculator, quiet=True)
                        period_returns.append(total_return)
                        print(f"    Run {run_id+1}: TR = {total_return:.4f}")
                    except Exception as e:
                        print(f"    Run failed: {e}")
                        continue
            else:
                print(f"  ✓ Loaded {len(period_returns)} precomputed results")
                
        else:
            period_returns = []
            
            for run_id in range(args.runs):
                try:
                    seed = RANDOM_SEEDS[run_id % len(RANDOM_SEEDS)]
                    total_return = run_single_strategy(
                        config, data_loader, lim_calculator, period, run_id, seed=seed
                    )
                    period_returns.append(total_return)
                except Exception as e:
                    print(f"  Run failed: {e}")
                    continue
        
        if period_returns:
            tr_data[period] = period_returns
            mean_tr = np.mean(period_returns)
            std_tr = np.std(period_returns)
            model_type = "multiseed precompute" if period == 30 else "fixed rebalance"
            print(f"  Summary ({model_type}): mean={mean_tr:.4f}, std={std_tr:.4f}, runs={len(period_returns)}")
        else:
            print(f"  ⚠️ All {period}-day runs failed!")
        print()
    
    
    if tr_data:
        plot_tr_curves(tr_data, output_dir)
        
        print("\n" + "=" * 60)
        print("TR curve analysis complete!")
        print("=" * 60)
        print(f"Results saved to: {output_dir}")
        print("Generated files:")
        print("- TR_curves_analysis.png: TR curve with confidence bands")
        print("- TR_analysis_summary.txt: statistics summary")
        
        print("\nExperiment summary:")
        if 30 in tr_data and len(tr_data[30]) > 1:
            mean_30 = np.mean(tr_data[30])
            std_30 = np.std(tr_data[30])
            print(f"- Main experiment (30-day, multiseed): mean={mean_30:.4f}, std={std_30:.4f} (n={len(tr_data[30])})")
        elif 30 in tr_data:
            mean_30 = np.mean(tr_data[30])
            print(f"- Main experiment (30-day, multiseed): TR = {mean_30:.4f} (single run)")
        else:
            print("- Main experiment (30-day): not loaded")
        print("- Other periods (fixed rebalance + pretrained model): show frequency impact")
        print("- All results include confidence intervals")
    else:
        print("Error: no strategy ran successfully")

if __name__ == '__main__':
    main() 

