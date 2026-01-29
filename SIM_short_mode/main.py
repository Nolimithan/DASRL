"""
SIM-enhanced RL strategy for semiconductor stock selection.

This strategy uses SIM (Intraday Loss Accumulation) with PPO for portfolio
optimization in a short-only setting. Higher sim* indicates greater potential
short-side returns over the evaluation window.
"""

import os
import argparse
import datetime
import yaml
import numpy as np
import pandas as pd
import warnings
import tensorflow as tf
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from utils.data_loader import DataLoader
from utils.sim_calculator import SIMCalculator
from models.ppo_agent import PPOAgent
from utils.portfolio import Portfolio
from utils.environment import TradingEnvironment
from visualization.visualizer import Visualizer
from config.config import get_config, create_default_config_file
from utils.data_loader_factory import create_data_loader

def parse_args():
    parser = argparse.ArgumentParser(description='SIM-enhanced reinforcement learning strategy')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'backtest'],
                        help='Run mode: train, test, or backtest')
    parser.add_argument('--config', type=str, default='config/default.yaml',
                        help='Config file path')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualizations')
    parser.add_argument('--create_config', action='store_true',
                        help='Create default config file')
    parser.add_argument('--ablation', action='store_true',
                        help='Run ablation studies')
    parser.add_argument('--start_date', type=str, 
                        help='Test start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, 
                        help='Test end date (YYYY-MM-DD)')
    parser.add_argument('--compare_stop_loss', action='store_true',
                        help='Compare stop-loss strategies')
    parser.add_argument('--optimize_stop_loss', action='store_true',
                        help='Optimize ATR stop-loss parameters')
    parser.add_argument('--group_type', type=str, default='TOP', choices=['TOP', 'MIDDLE', 'LOW'],
                        help='Stock group type: TOP, MIDDLE, LOW')
    parser.add_argument('--enable_proxy', action='store_true',
                        help='Enable yfinance proxy')
    parser.add_argument('--proxy', type=str, default='http://127.0.0.1:7890',
                        help='Proxy server address (default: http://127.0.0.1:7890)')
    return parser.parse_args()

def load_config(config_path):
    """Load configuration file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def setup_directories(config):
    """Ensure required directories exist."""
    os.makedirs(os.path.dirname(config['paths']['model_save_path']), exist_ok=True)
    
    os.makedirs(config['paths']['visualization_path'], exist_ok=True)

def train_agent(config, env, agent, epochs=100, steps_per_epoch=1000):
    """Train the PPO agent."""
    print(f"Starting PPO training: epochs={epochs}, steps_per_epoch={steps_per_epoch}")
    
    training_start_date = config['data']['start_date']
    training_end_date = config['data']['end_date']
    
    env.reset(training_start_date, training_end_date)
    
    agent.train(env, epochs=epochs, steps_per_epoch=steps_per_epoch)
    
    agent.save(config['paths']['model_save_path'])
    
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = config['paths'].get('results_path', 'results')
    train_results_dir = os.path.join(results_dir, f"train_results_{timestamp}")
    os.makedirs(train_results_dir, exist_ok=True)
    
    train_config = {
        'start_date': training_start_date,
        'end_date': training_end_date,
        'epochs': epochs,
        'steps_per_epoch': steps_per_epoch,
        'training_timestamp': timestamp,
    }
    config_path = os.path.join(train_results_dir, "train_config.yaml")
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(train_config, f, default_flow_style=False, allow_unicode=True)
    print(f"Training config saved to: {config_path}")
    
    metrics_df = pd.DataFrame({
        'episode': list(range(1, len(agent.training_metrics['actor_loss']) + 1)),
        'actor_loss': agent.training_metrics['actor_loss'],
        'critic_loss': agent.training_metrics['critic_loss'],
        'entropy': agent.training_metrics['entropy'],
        'total_rewards': agent.training_metrics['total_rewards']
    })
    metrics_path = os.path.join(train_results_dir, "training_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Training metrics saved to: {metrics_path}")
    
    print("Evaluating strategy on training data...")
    portfolio = Portfolio(
        initial_capital=config['portfolio']['initial_capital'],
        commission_rate=config['trading']['commission_rate']
    )
    
    env.reset(training_start_date, training_end_date)
    
    portfolio.run_strategy(agent, env, start_date=training_start_date, end_date=training_end_date)
    
    train_metrics = portfolio.get_performance_summary()
    
    train_history_path = os.path.join(train_results_dir, "train_history.csv")
    history_file = env.export_history(train_history_path)
    print(f"Training history exported to: {history_file}")
    
    metrics_path = os.path.join(train_results_dir, "performance_metrics.csv")
    pd.DataFrame([train_metrics]).to_csv(metrics_path, index=False)
    print(f"Training performance summary exported to: {metrics_path}")
    
    print("Generating training visualizations...")
    vis_path = os.path.join(train_results_dir, "visualization")
    os.makedirs(vis_path, exist_ok=True)
    
    original_vis_path = config['paths']['visualization_path']
    config['paths']['visualization_path'] = vis_path
    
    visualizer = Visualizer(output_dir=vis_path)
    visualizer.plot_results(portfolio)
    
    plt.figure(figsize=(12, 6))
    plt.plot(metrics_df['episode'], metrics_df['total_rewards'], marker='o', linestyle='-', color='#1f77b4')
    plt.title('Training Rewards per Episode', fontsize=14, fontweight='bold')
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Total Reward', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    rewards_path = os.path.join(vis_path, f'rewards_curve_{timestamp}.png')
    plt.savefig(rewards_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(metrics_df['episode'], metrics_df['actor_loss'], color='#2ca02c')
    plt.title('Actor Loss', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 1, 2)
    plt.plot(metrics_df['episode'], metrics_df['critic_loss'], color='#d62728')
    plt.title('Critic Loss', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 1, 3)
    plt.plot(metrics_df['episode'], metrics_df['entropy'], color='#9467bd')
    plt.title('Entropy', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    losses_path = os.path.join(vis_path, f'losses_curve_{timestamp}.png')
    plt.savefig(losses_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    config['paths']['visualization_path'] = original_vis_path
    print(f"Training visualizations saved to: {vis_path}")
    
    return agent

def run_trading_strategy(config, env, agent, portfolio):
    """Run trading strategy using a trained agent."""
    print("Running trading strategy")
    
    start_date = config['data']['start_date']
    end_date = config['data']['end_date']
    
    portfolio.run_strategy(agent, env, start_date=start_date, end_date=end_date)
    
    return portfolio

def run_ablation_studies(config, env, portfolio, data_loader, sim_calculator, agent):
    """Run ablation studies and comparisons."""
    print("Starting ablation studies")
    
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join(config['paths']['results_path'], f"ablation_study_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    plots_dir = os.path.join(results_dir, "plots")
    metrics_dir = os.path.join(results_dir, "metrics")
    models_dir = os.path.join(results_dir, "models")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    config_path = os.path.join(results_dir, "experiment_config.yaml")
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    start_date = config['data']['start_date']
    end_date = config['data']['end_date']
    
    all_results = {}
    
    ablation_config = {
        'baseline_agent': agent,
    }
    
    print("Running baseline strategy...")
    baseline_portfolio = Portfolio(
        initial_capital=config['portfolio']['initial_capital'],
        commission_rate=config['trading']['commission_rate']
    )
    baseline_results = baseline_portfolio.run_strategy(
        agent, env, start_date=start_date, end_date=end_date
    )
    all_results['baseline'] = baseline_results
    
    if config['ablation']['run_no_SIM']:
        print("Preparing no-SIM strategy...")
        no_SIM_env = TradingEnvironment(
            config=config,
            data_loader=data_loader,
            sim_calculator=None
        )
        no_SIM_agent = PPOAgent(
            state_dim=no_SIM_env.observation_space.shape[0],
            action_dim=no_SIM_env.action_space.shape[0],
            lr=config['rl']['learning_rate'],
            gamma=config['rl']['gamma'],
            clip_ratio=config['rl']['clip_ratio'],
            value_coef=config['rl']['value_coef'],
            entropy_coef=config['rl']['entropy_coef']
        )
        no_SIM_agent.load(config['paths']['model_load_path'])
        
        no_SIM_portfolio = Portfolio(
            initial_capital=config['portfolio']['initial_capital'],
            commission_rate=config['trading']['commission_rate']
        )
        
        no_SIM_results = no_SIM_portfolio.run_strategy(
            no_SIM_agent, no_SIM_env, start_date=start_date, end_date=end_date
        )
        
        ablation_config['no_SIM_agent'] = no_SIM_agent
        ablation_config['no_SIM_env'] = no_SIM_env
    
    if config['ablation']['run_fixed_30day']:
        print("Preparing fixed 30-day rebalance strategy...")
        fixed_config = config.copy()
        fixed_config['environment']['rebalance_period'] = 30
        fixed_30day_env = TradingEnvironment(
            config=fixed_config,
            data_loader=data_loader,
            sim_calculator=sim_calculator
        )
        
        fixed_30day_portfolio = Portfolio(
            initial_capital=config['portfolio']['initial_capital'],
            commission_rate=config['trading']['commission_rate']
        )
        
        fixed_30day_results = fixed_30day_portfolio.run_strategy(
            agent, fixed_30day_env, start_date=start_date, end_date=end_date
        )
        
        ablation_config['fixed_30day_agent'] = agent
        ablation_config['fixed_30day_env'] = fixed_30day_env
    
    if config['ablation']['run_no_cov_penalty']:
        print("Preparing no-covariance-penalty strategy...")
        no_cov_config = config.copy()
        if 'reward_params' in no_cov_config.get('rl', {}):
            no_cov_config['rl']['reward_params']['lambda_2'] = 0
        no_cov_penalty_env = TradingEnvironment(
            config=no_cov_config,
            data_loader=data_loader,
            sim_calculator=sim_calculator
        )
        ablation_config['no_cov_penalty_agent'] = agent
        ablation_config['no_cov_penalty_env'] = no_cov_penalty_env
    
    if config['ablation']['run_no_exit_rule']:
        print("Preparing no-stop-loss strategy...")
        no_exit_config = config.copy()
        no_exit_config['environment']['stop_loss_threshold'] = 1.0
        no_exit_rule_env = TradingEnvironment(
            config=no_exit_config,
            data_loader=data_loader,
            sim_calculator=sim_calculator
        )
        ablation_config['no_exit_rule_agent'] = agent
        ablation_config['no_exit_rule_env'] = no_exit_rule_env
    
    if config['ablation']['run_equal_weight']:
        print("Preparing equal-weight strategy...")
        equal_weight_env = TradingEnvironment(
            config=config,
            data_loader=data_loader,
            sim_calculator=sim_calculator
        )
        equal_weight_portfolio = Portfolio(
            initial_capital=config['portfolio']['initial_capital'],
            commission_rate=config['trading']['commission_rate']
        )
        equal_weight_env.reset(start_date=start_date, end_date=end_date)
        state = equal_weight_env.reset(start_date=start_date, end_date=end_date)
        done = False
        class EqualWeightAgent:
            def predict(self, state):
                n_stocks = equal_weight_env.action_space.shape[0]
                return np.ones(n_stocks) / n_stocks
        
        equal_weight_agent = EqualWeightAgent()
        equal_weight_results = equal_weight_portfolio.run_strategy(
            equal_weight_agent, equal_weight_env, start_date=start_date, end_date=end_date
        )
        ablation_config['equal_weight_results'] = equal_weight_results
    
    if config['ablation']['run_mean_var']:
        print("Preparing mean-variance strategy...")
        pass
    
    if config['ablation']['run_momentum']:
        print("Preparing momentum strategy...")
        momentum_env = TradingEnvironment(
            config=config,
            data_loader=data_loader,
            sim_calculator=sim_calculator
        )
        momentum_portfolio = Portfolio(
            initial_capital=config['portfolio']['initial_capital'],
            commission_rate=config['trading']['commission_rate']
        )
        momentum_env.reset(start_date=start_date, end_date=end_date)
        class MomentumAgent:
            def predict(self, state):
                n_stocks = momentum_env.action_space.shape[0]
                weights = np.random.random(n_stocks)
                return weights / np.sum(weights)
        
        momentum_agent = MomentumAgent()
        momentum_results = momentum_portfolio.run_strategy(
            momentum_agent, momentum_env, start_date=start_date, end_date=end_date
        )
        ablation_config['momentum_results'] = momentum_results
    
    if config['ablation']['run_literature_rl']:
        print("Preparing literature RL model...")
        pass
    
    def save_strategy_results(strategy_name, results, portfolio_instance):
        """Save results for a single strategy."""
        metrics_path = os.path.join(metrics_dir, f"{strategy_name}_metrics.csv")
        pd.DataFrame([results]).to_csv(metrics_path, index=False)
        
        portfolio_values = pd.DataFrame({
            'date': portfolio_instance.history['date'],
            'portfolio_value': portfolio_instance.history['portfolio_value']
        })
        portfolio_values.set_index('date', inplace=True)
        values_path = os.path.join(metrics_dir, f"{strategy_name}_portfolio_values.csv")
        portfolio_values.to_csv(values_path)
        
        trades_path = os.path.join(metrics_dir, f"{strategy_name}_trades.csv")
        trades_df = pd.DataFrame({
            'date': portfolio_instance.history['date'],
            'transactions': portfolio_instance.history['transactions'],
            'transaction_costs': portfolio_instance.history['transaction_costs']
        })
        trades_df.to_csv(trades_path, index=False)
        
        plt.figure(figsize=(12, 6))
        plt.plot(portfolio_values.index, portfolio_values['portfolio_value'])
        plt.title(f"{strategy_name} Portfolio Value")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value")
        plt.savefig(os.path.join(plots_dir, f"{strategy_name}_portfolio_value.png"))
        plt.close()
    
    save_strategy_results("baseline", baseline_results, baseline_portfolio)
    
    if config['ablation']['run_no_SIM']:
        print("Running no-SIM strategy...")
        save_strategy_results("no_SIM", no_SIM_results, no_SIM_portfolio)
        all_results['no_SIM'] = no_SIM_results
    
    if config['ablation']['run_fixed_30day']:
        print("Running fixed 30-day rebalance strategy...")
        save_strategy_results("fixed_30day", fixed_30day_results, fixed_30day_portfolio)
        all_results['fixed_30day'] = fixed_30day_results
    
    # Generate summary comparison plot
    plt.figure(figsize=(15, 8))
    for strategy_name, results in all_results.items():
        portfolio_values = pd.read_csv(
            os.path.join(metrics_dir, f"{strategy_name}_portfolio_values.csv"),
            index_col=0, parse_dates=True
        )
        plt.plot(portfolio_values.index, portfolio_values['portfolio_value'], label=strategy_name)
    
    plt.title("Strategy Comparison")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plots_dir, "strategies_comparison.png"))
    plt.close()
    
    summary_df = pd.DataFrame()
    for strategy_name in all_results.keys():
        metrics = pd.read_csv(os.path.join(metrics_dir, f"{strategy_name}_metrics.csv"))
        summary_df[strategy_name] = metrics.iloc[0]
    
    summary_df = summary_df.transpose()
    summary_path = os.path.join(results_dir, "summary_comparison.csv")
    summary_df.to_csv(summary_path)
    
    metrics_to_plot = ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    for i, metric in enumerate(metrics_to_plot):
        if metric in summary_df.columns:
            ax = axes[i]
            summary_df[metric].plot(kind='bar', ax=ax)
            ax.set_title(metric.replace('_', ' ').title())
            ax.grid(True, alpha=0.3)
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "metrics_comparison.png"))
    plt.close()
    
    experiment_info = {
        'timestamp': timestamp,
        'start_date': start_date,
        'end_date': end_date,
        'group_type': config['environment']['group_type'],
        'strategies_tested': list(all_results.keys())
    }
    
    info_path = os.path.join(results_dir, "experiment_info.yaml")
    with open(info_path, 'w', encoding='utf-8') as f:
        yaml.dump(experiment_info, f, default_flow_style=False, allow_unicode=True)
    
    print(f"Ablation results saved to: {results_dir}")
    return all_results

def visualize_results(config, portfolio, ablation_results=None):
    """Visualize results and generate reports."""
    print("Starting visualization")
    
    visualizer = Visualizer(output_dir=config['paths']['visualization_path'])
    
    visualizer.plot_results(portfolio)
    
    if ablation_results:
        visualizer.plot_comparison(ablation_results, metrics=['total_return', 'sharpe_ratio', 'max_drawdown'])

def main():
    args = parse_args()
    
    if args.create_config:
        create_default_config_file()
        print("Default config created. Exiting.")
        return
    
    if not os.path.exists(args.config):
        print(f"Config file not found: {args.config}")
        print("Creating default config file...")
        create_default_config_file()
        
    config = get_config(args.config)
    
    if args.start_date:
        config['data']['start_date'] = args.start_date
        print(f"Using CLI start date: {args.start_date}")
    
    if args.end_date:
        config['data']['end_date'] = args.end_date
        print(f"Using CLI end date: {args.end_date}")
    
    print(f"Starting SIM-enhanced RL strategy - mode: {args.mode}")
    start_time = datetime.datetime.now()
    
    print("Initializing data loader...")
    data_loader = create_data_loader(config, proxy=args.proxy, enable_proxy=args.enable_proxy)
    
    print("Loading stock data...")
    data_loader.load_data()
    
    print("Initializing SIM calculator...")
    sim_calculator = SIMCalculator(
        window_size=config['sim']['window_size'],
        lookback_days=config['sim'].get('lookback_days', 90)
    )
    
    print("Setting up trading environment...")
    
    if 'environment' not in config:
        config['environment'] = {
            'portfolio_size': config['portfolio']['size'],
            'initial_capital': config['portfolio']['initial_capital'],
            'commission_rate': config['trading']['commission_rate'],
            'stop_loss_threshold': 0.10,
            'rebalance_period': 30
        }
    
    if args.group_type:
        config['environment']['group_type'] = args.group_type
        print(f"Using CLI group type: {args.group_type}")
    
    env = TradingEnvironment(
        config=config,
        data_loader=data_loader,
        sim_calculator=sim_calculator
    )
    
    print("Initializing RL agent...")
    agent = PPOAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        lr=config['rl']['learning_rate'],
        gamma=config['rl']['gamma'],
        clip_ratio=config['rl']['clip_ratio'],
        value_coef=config['rl']['value_coef'],
        entropy_coef=config['rl']['entropy_coef']
    )
    
    if args.mode == 'train':
        print("Starting training...")
        agent = train_agent(
            config,
            env,
            agent,
            epochs=config['rl']['epochs'],
            steps_per_epoch=config['rl']['steps_per_epoch']
        )
        print(f"Training complete. Model saved to {config['paths']['model_save_path']}")
        
        portfolio = Portfolio(
            initial_capital=config['portfolio']['initial_capital'],
            commission_rate=config['trading']['commission_rate']
        )
        ablation_results = None
    
    elif args.mode == 'test':
        print("Starting testing...")
        agent.load(config['paths']['model_load_path'])
        portfolio = Portfolio(
            initial_capital=config['portfolio']['initial_capital'],
            commission_rate=config['trading']['commission_rate']
        )
        start_date = config['data']['start_date']
        end_date = config['data']['end_date']
        print(f"Loading data from {start_date} to {end_date}...")
        
        portfolio = run_trading_strategy(config, env, agent, portfolio)
        print("Testing complete")
        
        metrics = portfolio.get_performance_summary()
        print("\nPerformance summary:")
        for key, value in metrics.items():
            print(f"{key}: {value}")
        
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = config['paths'].get('results_path', 'results')
        results_timestamp_dir = os.path.join(results_dir, f"test_results_{timestamp}")
        os.makedirs(results_timestamp_dir, exist_ok=True)
        
        test_config = {
            'start_date': start_date,
            'end_date': end_date,
            'portfolio_size': config['portfolio']['size'],
            'initial_capital': config['portfolio']['initial_capital'],
            'commission_rate': config['trading']['commission_rate'],
            'symbols_count': len(config['data']['symbols']),
            'test_timestamp': timestamp,
            'command_args': vars(args)
        }
        config_path = os.path.join(results_timestamp_dir, "test_config.yaml")
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(test_config, f, default_flow_style=False, allow_unicode=True)
        print(f"Test config saved to: {config_path}")
        
        results_path = os.path.join(results_timestamp_dir, "test_history.csv")
        history_file = env.export_history(results_path)
        print(f"\nTest history exported to: {history_file}")
        
        metrics_path = os.path.join(results_timestamp_dir, "performance_metrics.csv")
        pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
        print(f"Performance summary exported to: {metrics_path}")
        
        if args.ablation:
            print("Running ablation studies...")
            ablation_results = run_ablation_studies(config, env, portfolio, data_loader, sim_calculator, agent)
            ablation_path = os.path.join(results_timestamp_dir, "ablation_results.csv")
            pd.DataFrame(ablation_results).to_csv(ablation_path, index=False)
            print(f"Ablation results exported to: {ablation_path}")
            print("Ablation studies complete")
        else:
            ablation_results = None
            
        if args.compare_stop_loss:
            print("\nRunning stop-loss comparison...")
            from utils.stop_loss_tester import ATRStopLossTester, compare_stop_loss_strategies
            
            stop_loss_results = compare_stop_loss_strategies(
                config, data_loader, sim_calculator, agent
            )
            stop_loss_path = os.path.join(results_timestamp_dir, "stop_loss_comparison.csv")
            pd.DataFrame(stop_loss_results).to_csv(stop_loss_path, index=False)
            print(f"Stop-loss comparison exported to: {stop_loss_path}")
            print("Stop-loss comparison complete")
        
        if args.optimize_stop_loss:
            print("\nRunning ATR stop-loss parameter optimization...")
            from utils.stop_loss_tester import run_parameter_optimization
            
            parameter_sets = [
                {'atr_window': 14, 'volatility_range': (0.5, 2.0), 'trend_adjustment': True},
                {'atr_window': 10, 'volatility_range': (0.6, 1.8), 'trend_adjustment': True},
                {'atr_window': 20, 'volatility_range': (0.7, 1.5), 'trend_adjustment': False},
                {'atr_window': 5, 'volatility_range': (0.8, 1.3), 'trend_adjustment': True},
                {'atr_window': 30, 'volatility_range': (0.4, 2.5), 'trend_adjustment': True}
            ]
            
            optimization_results = run_parameter_optimization(
                config, data_loader, sim_calculator, agent, parameter_sets
            )
            optimization_path = os.path.join(results_timestamp_dir, "atr_optimization.csv")
            pd.DataFrame(optimization_results).to_csv(optimization_path, index=False)
            print(f"ATR optimization exported to: {optimization_path}")
            print("ATR optimization complete")
    
    elif args.mode == 'backtest':
        print("Starting backtest and comparison...")
        
        print("Loading trained model...")
        agent.load(config['paths']['model_load_path'])
        
        portfolio = Portfolio(
            initial_capital=config['portfolio']['initial_capital'],
            commission_rate=config['trading']['commission_rate']
        )
        
        ablation_results = run_ablation_studies(config, env, portfolio, data_loader, sim_calculator, agent)
        
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = config['paths'].get('results_path', 'results')
        results_timestamp_dir = os.path.join(results_dir, f"backtest_results_{timestamp}")
        os.makedirs(results_timestamp_dir, exist_ok=True)
        
        backtest_config = {
            'start_date': config['data']['start_date'],
            'end_date': config['data']['end_date'],
            'portfolio_size': config['portfolio']['size'],
            'initial_capital': config['portfolio']['initial_capital'],
            'commission_rate': config['trading']['commission_rate'],
            'symbols_count': len(config['data']['symbols']),
            'backtest_timestamp': timestamp,
            'command_args': vars(args)
        }
        config_path = os.path.join(results_timestamp_dir, "backtest_config.yaml")
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(backtest_config, f, default_flow_style=False, allow_unicode=True)
        print(f"Backtest config saved to: {config_path}")
        
        if isinstance(ablation_results, dict) and 'error' not in ablation_results:
            ablation_path = os.path.join(results_timestamp_dir, "ablation_results.csv")
            try:
                pd.DataFrame(ablation_results).to_csv(ablation_path, index=False)
                print(f"Ablation results exported to: {ablation_path}")
            except Exception as e:
                print(f"Error exporting ablation results: {str(e)}")
                error_path = os.path.join(results_timestamp_dir, "ablation_error.txt")
                with open(error_path, 'w', encoding='utf-8') as f:
                    f.write(f"Error: {str(e)}\n")
                    f.write(f"Results: {str(ablation_results)}")
        else:
            error_path = os.path.join(results_timestamp_dir, "ablation_error.txt")
            with open(error_path, 'w', encoding='utf-8') as f:
                if isinstance(ablation_results, dict) and 'error' in ablation_results:
                    f.write(f"Error: {ablation_results['error']}\n")
                else:
                    f.write(f"Error: Unknown error\n")
                    f.write(f"Results: {str(ablation_results)}")
        
        if args.visualize and isinstance(ablation_results, dict) and 'error' not in ablation_results:
            vis_path = os.path.join(results_timestamp_dir, "visualization")
            os.makedirs(vis_path, exist_ok=True)
            original_vis_path = config['paths']['visualization_path']
            config['paths']['visualization_path'] = vis_path
            
            visualizer = Visualizer(output_dir=vis_path)
            visualizer.plot_comparison(ablation_results, metrics=['total_return', 'sharpe_ratio', 'max_drawdown'])
            
            config['paths']['visualization_path'] = original_vis_path
            print(f"Visualizations saved to: {vis_path}")
            
        print("Backtest and comparison complete")
    
    if args.visualize:
        print("Generating visualizations...")
        
        if args.mode == 'test':
            vis_path = os.path.join(results_timestamp_dir, "visualization")
            os.makedirs(vis_path, exist_ok=True)
            original_vis_path = config['paths']['visualization_path']
            config['paths']['visualization_path'] = vis_path
            
            visualize_results(config, portfolio, ablation_results)
            
            config['paths']['visualization_path'] = original_vis_path
            print(f"Visualizations saved to: {vis_path}")
        else:
            os.makedirs(config['paths']['visualization_path'], exist_ok=True)
            visualize_results(config, portfolio, ablation_results)
    
    end_time = datetime.datetime.now()
    run_time = end_time - start_time
    print(f"\nTotal runtime: {run_time}")
    print("Execution complete")

if __name__ == '__main__':
    main() 

