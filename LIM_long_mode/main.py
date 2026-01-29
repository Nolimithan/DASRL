"""
LIM-enhanced reinforcement learning strategy - semiconductor stock selection
Main entry point

This strategy uses LIM (Intraday Return Accumulation) with PPO to optimize a stock portfolio.
LIM measures a stock's short-term return potential as follows:

1. Intraday return accumulation based on open/close prices:
   - Buy at market open and sell at close
   - Include transaction cost (F)
   - Accumulate only profitable trades (no shorting)

2. Calculation steps:
   - Single intraday trade wealth change: Wt = W0 * [Ct*(1-F)²/Ot] (if Ct*(1-F)²/Ot ≥ 1)
                                      Wt = W0 (if Ct*(1-F)²/Ot < 1)
   - Accumulate profitable trades over a period
   - Normalize: lim* = (LIMi/LIMmax)*100%

3. A higher lim* indicates stronger return potential over the next period
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

# Ignore TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Set TensorFlow log level
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from utils.data_loader import DataLoader
from utils.lim_calculator import LIMCalculator
from models.ppo_agent import PPOAgent
from utils.portfolio import Portfolio
from utils.environment import TradingEnvironment
from visualization.visualizer import Visualizer
from config.config import get_config, create_default_config_file
from utils.data_loader_factory import create_data_loader

def parse_args():
    parser = argparse.ArgumentParser(description='LIM-enhanced reinforcement learning strategy')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'backtest'],
                        help='Run mode: train, test, or backtest')
    parser.add_argument('--config', type=str, default='config/default.yaml',
                        help='Config file path')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualizations')
    parser.add_argument('--create_config', action='store_true',
                        help='Create the default config file')
    parser.add_argument('--ablation', action='store_true',
                        help='Run ablation studies')
    parser.add_argument('--start_date', type=str, 
                        help='Test start date (format: YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, 
                        help='Test end date (format: YYYY-MM-DD)')
    # Stock group parameter
    parser.add_argument('--group_type', type=str, default='TOP', choices=['TOP', 'MIDDLE', 'LOW'],
                        help='Stock group type: TOP (top), MIDDLE (middle), LOW (bottom)')
    # Proxy parameters
    parser.add_argument('--enable_proxy', action='store_true',
                        help='Enable yfinance proxy')
    parser.add_argument('--proxy', type=str, default='http://127.0.0.1:7890',
                        help='Proxy server address (default: http://127.0.0.1:7890)')
    return parser.parse_args()

def load_config(config_path):
    """Load config file
    
    Args:
        config_path: Config file path
        
    Returns:
        dict: Config dictionary
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def setup_directories(config):
    """Set required directories
    
    Args:
        config: Config dictionary
    """
    # Ensure model save directory exists
    os.makedirs(os.path.dirname(config['paths']['model_save_path']), exist_ok=True)
    
    # Ensure visualization directory exists
    os.makedirs(config['paths']['visualization_path'], exist_ok=True)

def train_agent(config, env, agent, epochs=100, steps_per_epoch=1000):
    """Train the RL agent
    
    Args:
        config: Config dictionary
        env: Trading environment instance
        agent: PPO agent instance
        epochs: Training epochs
        steps_per_epoch: Steps per epoch
        
    Returns:
        agent: Trained agent instance
    """
    print(f"Starting PPO training, epochs: {epochs}, steps per epoch: {steps_per_epoch}")
    
    # Set training parameters
    training_start_date = config['data']['start_date']
    training_end_date = config['data']['end_date']
    
    # Reset environment with trading date range
    env.reset(training_start_date, training_end_date)
    
    # Train agent
    agent.train(env, epochs=epochs, steps_per_epoch=steps_per_epoch)
    
    # Save trained model
    agent.save(config['paths']['model_save_path'])
    
    # Create timestamped training results folder
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = config['paths'].get('results_path', 'results')
    train_results_dir = os.path.join(results_dir, f"train_results_{timestamp}")
    os.makedirs(train_results_dir, exist_ok=True)
    
    # Save training config
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
    
    # Collect and save training metrics
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
    
    # Evaluate strategy on training data
    print("Running strategy evaluation on training data...")
    # Create portfolio for training evaluation
    portfolio = Portfolio(
        initial_capital=config['portfolio']['initial_capital'],
        commission_rate=config['trading']['commission_rate']
    )
    
    # Reset environment to run on training data
    env.reset(training_start_date, training_end_date)
    
    # Run strategy and record results
    portfolio.run_strategy(agent, env, start_date=training_start_date, end_date=training_end_date)
    
    # Get performance metrics
    train_metrics = portfolio.get_performance_summary()
    
    # Export training history
    train_history_path = os.path.join(train_results_dir, "train_history.csv")
    history_file = env.export_history(train_history_path)
    print(f"Training history exported to: {history_file}")
    
    # Export performance summary
    metrics_path = os.path.join(train_results_dir, "performance_metrics.csv")
    pd.DataFrame([train_metrics]).to_csv(metrics_path, index=False)
    print(f"Training performance summary exported to: {metrics_path}")
    
    # Visualize training results
    print("Generating training visualizations...")
    # Create visualization folder
    vis_path = os.path.join(train_results_dir, "visualization")
    os.makedirs(vis_path, exist_ok=True)
    
    # Temporarily save original visualization path
    original_vis_path = config['paths']['visualization_path']
    # Override visualization path in config
    config['paths']['visualization_path'] = vis_path
    
    # Use visualizer to generate plots
    visualizer = Visualizer(output_dir=vis_path)
    visualizer.plot_results(portfolio)
    
    # Plot reward curve
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
    
    # Plot loss curves
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
    
    # Restore original visualization path
    config['paths']['visualization_path'] = original_vis_path
    print(f"Training visualizations saved to: {vis_path}")
    
    return agent

def run_trading_strategy(config, env, agent, portfolio):
    """Run the trading strategy
    
    Run simulated trading with the trained agent
    
    Args:
        config: Config dictionary
        env: Trading environment instance
        agent: PPO agent instance
        portfolio: Portfolio instance
        
    Returns:
        portfolio: Updated portfolio instance
    """
    print("Starting trading strategy")
    
    # Get date range
    start_date = config['data']['start_date']
    end_date = config['data']['end_date']
    
    # Run strategy and record results
    portfolio.run_strategy(agent, env, start_date=start_date, end_date=end_date)
    
    return portfolio

def run_ablation_studies(config, env, portfolio, data_loader, lim_calculator, agent):
    """Run ablation studies
    
    Evaluate different strategy settings and store results in a timestamped folder
    
    Args:
        config: Config dictionary
        env: Trading environment instance
        portfolio: Portfolio instance
        data_loader: Data loader instance
        lim_calculator: LIM calculator instance
        agent: RL agent
        
    Returns:
        dict: Ablation results
    """
    print("Starting ablation studies")
    
    # Create timestamped results folder
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join(config['paths']['results_path'], f"ablation_study_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Create subfolders
    plots_dir = os.path.join(results_dir, "plots")
    metrics_dir = os.path.join(results_dir, "metrics")
    models_dir = os.path.join(results_dir, "models")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # Save experiment config
    config_path = os.path.join(results_dir, "experiment_config.yaml")
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    # Get date range
    start_date = config['data']['start_date']
    end_date = config['data']['end_date']
    
    # Store all experiment results
    all_results = {}
    
    # Configure ablation
    ablation_config = {
        'baseline_agent': agent,  # Use the passed agent as baseline
    }
    
    # Run baseline strategy and save results
    print("Running baseline strategy...")
    baseline_portfolio = Portfolio(
        initial_capital=config['portfolio']['initial_capital'],
        commission_rate=config['trading']['commission_rate']
    )
    baseline_results = baseline_portfolio.run_strategy(
        agent, env, start_date=start_date, end_date=end_date
    )
    all_results['baseline'] = baseline_results
    
    # Check whether to run no-LIM strategy
    if config['ablation']['run_no_LIM']:
        print("Preparing no-LIM strategy...")
        # Create an environment without LIM
        from utils.environment import TradingEnvironment
        no_LIM_env = TradingEnvironment(
            config=config,
            data_loader=data_loader,
            lim_calculator=None  # Disable LIM by passing None
        )
        # Create a new agent for the no-LIM environment
        from models.ppo_agent import PPOAgent
        no_LIM_agent = PPOAgent(
            state_dim=no_LIM_env.observation_space.shape[0],
            action_dim=no_LIM_env.action_space.shape[0],
            lr=config['rl']['learning_rate'],
            gamma=config['rl']['gamma'],
            clip_ratio=config['rl']['clip_ratio'],
            value_coef=config['rl']['value_coef'],
            entropy_coef=config['rl']['entropy_coef']
        )
        # Load trained model
        no_LIM_agent.load(config['paths']['model_load_path'])
        
        # Create portfolio for no-LIM strategy
        no_LIM_portfolio = Portfolio(
            initial_capital=config['portfolio']['initial_capital'],
            commission_rate=config['trading']['commission_rate']
        )
        
        # Run no-LIM strategy
        no_LIM_results = no_LIM_portfolio.run_strategy(
            no_LIM_agent, no_LIM_env, start_date=start_date, end_date=end_date
        )
        
        ablation_config['no_LIM_agent'] = no_LIM_agent
        ablation_config['no_LIM_env'] = no_LIM_env
    
    # Check whether to run fixed 30-day rebalance strategy
    if config['ablation']['run_fixed_30day']:
        print("Preparing fixed 30-day rebalance strategy...")
        # Create environment with fixed 30-day rebalance
        from utils.environment import TradingEnvironment
        fixed_config = config.copy()
        fixed_config['environment']['rebalance_period'] = 30
        fixed_30day_env = TradingEnvironment(
            config=fixed_config,
            data_loader=data_loader,
            lim_calculator=lim_calculator
        )
        
        # Create portfolio for fixed 30-day strategy
        fixed_30day_portfolio = Portfolio(
            initial_capital=config['portfolio']['initial_capital'],
            commission_rate=config['trading']['commission_rate']
        )
        
        # Run fixed 30-day strategy
        fixed_30day_results = fixed_30day_portfolio.run_strategy(
            agent, fixed_30day_env, start_date=start_date, end_date=end_date
        )
        
        # Reuse the same agent
        ablation_config['fixed_30day_agent'] = agent
        ablation_config['fixed_30day_env'] = fixed_30day_env
    
    # Check whether to run no-covariance-penalty strategy
    if config['ablation']['run_no_cov_penalty']:
        print("Preparing no-covariance-penalty strategy...")
        # Create environment without covariance penalty
        from utils.environment import TradingEnvironment
        no_cov_config = config.copy()
        if 'reward_params' in no_cov_config.get('rl', {}):
            no_cov_config['rl']['reward_params']['lambda_2'] = 0  # Set covariance penalty weight to 0
        no_cov_penalty_env = TradingEnvironment(
            config=no_cov_config,
            data_loader=data_loader,
            lim_calculator=lim_calculator
        )
        # Reuse the same agent
        ablation_config['no_cov_penalty_agent'] = agent
        ablation_config['no_cov_penalty_env'] = no_cov_penalty_env
    
    # Check whether to run no-stop-loss rule strategy
    if config['ablation']['run_no_exit_rule']:
        print("Preparing no-stop-loss strategy...")
        # Create environment without stop-loss rule
        from utils.environment import TradingEnvironment
        no_exit_config = config.copy()
        no_exit_config['environment']['stop_loss_threshold'] = 1.0  # Set very high threshold to disable stop-loss
        no_exit_rule_env = TradingEnvironment(
            config=no_exit_config,
            data_loader=data_loader,
            lim_calculator=lim_calculator
        )
        # Reuse the same agent
        ablation_config['no_exit_rule_agent'] = agent
        ablation_config['no_exit_rule_env'] = no_exit_rule_env
    
    # Check whether to run equal-weight strategy
    if config['ablation']['run_equal_weight']:
        print("Preparing equal-weight strategy...")
        # Implement equal-weight strategy
        from utils.environment import TradingEnvironment
        equal_weight_env = TradingEnvironment(
            config=config,
            data_loader=data_loader,
            lim_calculator=lim_calculator
        )
        # Create portfolio to run equal-weight strategy
        equal_weight_portfolio = Portfolio(
            initial_capital=config['portfolio']['initial_capital'],
            commission_rate=config['trading']['commission_rate']
        )
        # Reset environment
        equal_weight_env.reset(start_date=start_date, end_date=end_date)
        # Simple state tracking for equal-weight strategy
        state = equal_weight_env.reset(start_date=start_date, end_date=end_date)
        done = False
        # Create an equal-weight agent
        class EqualWeightAgent:
            def predict(self, state):
                # Return equal-weight allocation
                n_stocks = equal_weight_env.action_space.shape[0]
                return np.ones(n_stocks) / n_stocks
        
        equal_weight_agent = EqualWeightAgent()
        equal_weight_results = equal_weight_portfolio.run_strategy(
            equal_weight_agent, equal_weight_env, start_date=start_date, end_date=end_date
        )
        ablation_config['equal_weight_results'] = equal_weight_results
    
    # Check whether to run mean-variance optimization strategy
    if config['ablation']['run_mean_var']:
        print("Preparing mean-variance optimization strategy...")
        # Planned in Portfolio.run_ablation_studies
        # Not implemented here due to complexity
        pass
    
    # Check whether to run momentum strategy
    if config['ablation']['run_momentum']:
        print("Preparing momentum strategy...")
        # Implement momentum strategy
        from utils.environment import TradingEnvironment
        momentum_env = TradingEnvironment(
            config=config,
            data_loader=data_loader,
            lim_calculator=lim_calculator
        )
        # Create portfolio to run momentum strategy
        momentum_portfolio = Portfolio(
            initial_capital=config['portfolio']['initial_capital'],
            commission_rate=config['trading']['commission_rate']
        )
        # Reset environment
        momentum_env.reset(start_date=start_date, end_date=end_date)
        # Simple momentum agent
        class MomentumAgent:
            def predict(self, state):
                # Use past returns as weights
                # Assume the state contains past return information
                # Simplified implementation; should align with environment state
                n_stocks = momentum_env.action_space.shape[0]
                # Generate random weights; should be based on returns in practice
                weights = np.random.random(n_stocks)
                # Normalize weights
                return weights / np.sum(weights)
        
        momentum_agent = MomentumAgent()
        momentum_results = momentum_portfolio.run_strategy(
            momentum_agent, momentum_env, start_date=start_date, end_date=end_date
        )
        ablation_config['momentum_results'] = momentum_results
    
    # Check whether to run literature RL model
    if config['ablation']['run_literature_rl']:
        print("Preparing literature RL model...")
        # Planned in Portfolio.run_ablation_studies
        # Not implemented here due to complexity
        pass
    
    # Save results after each strategy
    def save_strategy_results(strategy_name, results, portfolio_instance):
        """Save results for a single strategy."""
        # Save performance metrics
        metrics_path = os.path.join(metrics_dir, f"{strategy_name}_metrics.csv")
        pd.DataFrame([results]).to_csv(metrics_path, index=False)
        
        # Save portfolio value curve
        portfolio_values = pd.DataFrame({
            'date': portfolio_instance.history['date'],
            'portfolio_value': portfolio_instance.history['portfolio_value']
        })
        portfolio_values.set_index('date', inplace=True)
        values_path = os.path.join(metrics_dir, f"{strategy_name}_portfolio_values.csv")
        portfolio_values.to_csv(values_path)
        
        # Save trade records
        trades_path = os.path.join(metrics_dir, f"{strategy_name}_trades.csv")
        trades_df = pd.DataFrame({
            'date': portfolio_instance.history['date'],
            'transactions': portfolio_instance.history['transactions'],
            'transaction_costs': portfolio_instance.history['transaction_costs']
        })
        trades_df.to_csv(trades_path, index=False)
        
        # Generate strategy performance plot
        plt.figure(figsize=(12, 6))
        plt.plot(portfolio_values.index, portfolio_values['portfolio_value'])
        plt.title(f"{strategy_name} Portfolio Value")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value")
        plt.savefig(os.path.join(plots_dir, f"{strategy_name}_portfolio_value.png"))
        plt.close()
    
    # Save baseline strategy results
    save_strategy_results("baseline", baseline_results, baseline_portfolio)
    
    # Run other strategies and save results
    if config['ablation']['run_no_LIM']:
        print("Running no-LIM strategy...")
        save_strategy_results("no_LIM", no_LIM_results, no_LIM_portfolio)
        all_results['no_LIM'] = no_LIM_results
    
    if config['ablation']['run_fixed_30day']:
        print("Running fixed 30-day rebalance strategy...")
        save_strategy_results("fixed_30day", fixed_30day_results, fixed_30day_portfolio)
        all_results['fixed_30day'] = fixed_30day_results
    
    # ... [Other strategies and result saving] ...
    
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
    
    # Generate summary report
    summary_df = pd.DataFrame()
    for strategy_name in all_results.keys():
        metrics = pd.read_csv(os.path.join(metrics_dir, f"{strategy_name}_metrics.csv"))
        summary_df[strategy_name] = metrics.iloc[0]
    
    # Add strategy comparison metrics
    summary_df = summary_df.transpose()  # Transpose for better display
    summary_path = os.path.join(results_dir, "summary_comparison.csv")
    summary_df.to_csv(summary_path)
    
    # Generate strategy comparison charts
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
    
    # Save experiment info
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
    """Visualize results
    
    Create result charts and analysis report
    
    Args:
        config: Config dictionary
        portfolio: Portfolio instance
        ablation_results: Ablation results dict
    """
    print("Starting visualization")
    
    # Initialize visualizer
    visualizer = Visualizer(output_dir=config['paths']['visualization_path'])
    
    # Plot main strategy results
    visualizer.plot_results(portfolio)
    
    # Visualize ablation results if provided
    if ablation_results:
        # Create comparison charts
        visualizer.plot_comparison(ablation_results, metrics=['total_return', 'sharpe_ratio', 'max_drawdown'])

def main():
    # Parse CLI arguments
    args = parse_args()
    
    # Create default config file if requested
    if args.create_config:
        create_default_config_file()
        print("Default config file created. Exiting.")
        return
    
    # Ensure config file exists
    if not os.path.exists(args.config):
        print(f"Config file not found: {args.config}")
        print("Creating default config file...")
        create_default_config_file()
        
    # Load config
    config = get_config(args.config)
    
    # Override dates from CLI if provided
    if args.start_date:
        config['data']['start_date'] = args.start_date
        print(f"Using CLI start date: {args.start_date}")
    
    if args.end_date:
        config['data']['end_date'] = args.end_date
        print(f"Using CLI end date: {args.end_date}")
    
    print(f"Starting LIM-enhanced RL strategy - mode: {args.mode}")
    start_time = datetime.datetime.now()
    
    # Initialize data loader
    print("Initializing data loader...")
    data_loader = create_data_loader(config, proxy=args.proxy, enable_proxy=args.enable_proxy)
    
    # Load stock data
    print("Loading stock data...")
    data_loader.load_data()
    
    # Initialize LIM calculator
    print("Initializing LIM calculator...")
    lim_calculator = LIMCalculator(
        window_size=config['lim']['window_size'],
        lookback_days=config['lim'].get('lookback_days', 90)  # Read from config, default 90 days
    )
    
    # Set trading environment
    print("Setting trading environment...")
    
    # Ensure config has environment section
    if 'environment' not in config:
        config['environment'] = {
            'portfolio_size': config['portfolio']['size'],
            'initial_capital': config['portfolio']['initial_capital'],
            'commission_rate': config['trading']['commission_rate'],
            'stop_loss_threshold': 0.10,
            'rebalance_period': 30
        }
    
    # Apply CLI group_type to environment config
    if args.group_type:
        config['environment']['group_type'] = args.group_type
        print(f"Using CLI stock group: {args.group_type}")
    
    env = TradingEnvironment(
        config=config,
        data_loader=data_loader,
        lim_calculator=lim_calculator
    )
    
    # Initialize RL agent
    print("Initializing RL agent...")
    agent = PPOAgent(
        state_dim=env.observation_space.shape[0],  # Use actual state dimension
        action_dim=env.action_space.shape[0],      # Use actual action dimension
        lr=config['rl']['learning_rate'],
        gamma=config['rl']['gamma'],
        clip_ratio=config['rl']['clip_ratio'],
        value_coef=config['rl']['value_coef'],
        entropy_coef=config['rl']['entropy_coef']
    )
    
    # Run based on mode
    if args.mode == 'train':
        # Training mode
        print("Starting training...")
        agent = train_agent(
            config,
            env,
            agent,
            epochs=config['rl']['epochs'],
            steps_per_epoch=config['rl']['steps_per_epoch']
        )
        print(f"Training complete. Saved to {config['paths']['model_save_path']}")
        
        # Initialize portfolio and ablation_results for consistency
        portfolio = Portfolio(
            initial_capital=config['portfolio']['initial_capital'],
            commission_rate=config['trading']['commission_rate']
        )
        ablation_results = None
    
    elif args.mode == 'test':
        # Test mode - forward test with trained model
        print("Starting testing...")
        agent.load(config['paths']['model_load_path'])
        portfolio = Portfolio(
            initial_capital=config['portfolio']['initial_capital'],
            commission_rate=config['trading']['commission_rate']
        )
        # Use date range from config (may be overridden)
        start_date = config['data']['start_date']
        end_date = config['data']['end_date']
        print(f"Loading data from {start_date} to {end_date}...")
        
        portfolio = run_trading_strategy(config, env, agent, portfolio)
        print("Testing complete")
        
        # Print performance metrics
        metrics = portfolio.get_performance_summary()
        print("\nPerformance summary:")
        for key, value in metrics.items():
            print(f"{key}: {value}")
        
        # Create timestamped results folder
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = config['paths'].get('results_path', 'results')
        # Create timestamped subfolder under results
        results_timestamp_dir = os.path.join(results_dir, f"test_results_{timestamp}")
        # Ensure timestamp folder exists
        os.makedirs(results_timestamp_dir, exist_ok=True)
        
        # Save test config
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
        
        # Export test history to timestamp folder
        results_path = os.path.join(results_timestamp_dir, "test_history.csv")
        history_file = env.export_history(results_path)
        print(f"\nTest history exported to: {history_file}")
        
        # Export performance summary to timestamp folder
        metrics_path = os.path.join(results_timestamp_dir, "performance_metrics.csv")
        pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
        print(f"Performance summary exported to: {metrics_path}")
        
        # Run ablation studies if requested
        if args.ablation:
            print("Running ablation studies...")
            ablation_results = run_ablation_studies(config, env, portfolio, data_loader, lim_calculator, agent)
            # Export ablation results
            ablation_path = os.path.join(results_timestamp_dir, "ablation_results.csv")
            pd.DataFrame(ablation_results).to_csv(ablation_path, index=False)
            print(f"Ablation results exported to: {ablation_path}")
            print("Ablation studies complete")
        else:
            ablation_results = None
            
    
    elif args.mode == 'backtest':
        # Backtest mode - comparative experiments
        print("Starting backtest and comparison...")
        
        # Load trained model for comparison
        print("Loading trained model...")
        agent.load(config['paths']['model_load_path'])
        
        portfolio = Portfolio(
            initial_capital=config['portfolio']['initial_capital'],
            commission_rate=config['trading']['commission_rate']
        )
        
        # Run backtest and comparison
        ablation_results = run_ablation_studies(config, env, portfolio, data_loader, lim_calculator, agent)
        
        # Create timestamped results folder
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = config['paths'].get('results_path', 'results')
        results_timestamp_dir = os.path.join(results_dir, f"backtest_results_{timestamp}")
        os.makedirs(results_timestamp_dir, exist_ok=True)
        
        # Save backtest config
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
        
        # Export backtest results
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
        
        # If visualization is requested, create a visualization folder
        if args.visualize and isinstance(ablation_results, dict) and 'error' not in ablation_results:
            vis_path = os.path.join(results_timestamp_dir, "visualization")
            os.makedirs(vis_path, exist_ok=True)
            # Temporarily save original visualization path
            original_vis_path = config['paths']['visualization_path']
            # Override visualization path in config
            config['paths']['visualization_path'] = vis_path
            
            # Use visualizer to generate comparison charts
            visualizer = Visualizer(output_dir=vis_path)
            visualizer.plot_comparison(ablation_results, metrics=['total_return', 'sharpe_ratio', 'max_drawdown'])
            
            # Restore original visualization path
            config['paths']['visualization_path'] = original_vis_path
            print(f"Visualizations saved to: {vis_path}")
            
        print("Backtest and comparison complete")
    
    # Generate visualizations
    if args.visualize:
        print("Generating visualizations...")
        
        # In test mode, use timestamped folder for visualization output
        if args.mode == 'test':
            vis_path = os.path.join(results_timestamp_dir, "visualization")
            os.makedirs(vis_path, exist_ok=True)
            # Temporarily save original visualization path
            original_vis_path = config['paths']['visualization_path']
            # Override visualization path in config
            config['paths']['visualization_path'] = vis_path
            
            # Generate visualizations
            visualize_results(config, portfolio, ablation_results)
            
            # Restore original visualization path
            config['paths']['visualization_path'] = original_vis_path
            print(f"Visualizations saved to: {vis_path}")
        else:
            # Ensure visualization output directory exists
            os.makedirs(config['paths']['visualization_path'], exist_ok=True)
            visualize_results(config, portfolio, ablation_results)
    
    end_time = datetime.datetime.now()
    run_time = end_time - start_time
    print(f"\nTotal runtime: {run_time}")
    print("Execution complete")

if __name__ == '__main__':
    main() 

