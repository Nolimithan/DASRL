"""
Ablation study for Sharpe/Sortino selection.

Compares LIM against traditional risk-adjusted indicators.
"""

import os
import numpy as np
import pandas as pd
import yaml
from datetime import datetime
from utils.environment import TradingEnvironment
from models.ppo_agent import PPOAgent as PPOModel
from utils.data_loader import DataLoader
from utils.lim_calculator import LIMCalculator
from utils.portfolio import Portfolio
import warnings
warnings.filterwarnings('ignore')


def load_config(config_path='config/default.yaml'):
    """Load configuration file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def calculate_sharpe_ratio(stock_data, lookback_days=90, risk_free_rate=0.04):
    """Compute Sharpe Ratio for ranking."""
    try:
        if len(stock_data) < 10:
            return 0.0
        
        # Get close prices
        if 'close' in stock_data.columns:
            close_prices = stock_data['close'].values
        elif 'Close' in stock_data.columns:
            close_prices = stock_data['Close'].values
        else:
            return 0.0
        
        # Compute daily returns
        daily_returns = np.diff(close_prices) / close_prices[:-1]
        daily_returns = daily_returns[~np.isnan(daily_returns)]
        
        if len(daily_returns) < 5:
            return 0.0
        
        # Compute annualized return and volatility
        mean_return = np.mean(daily_returns) * 252
        std_return = np.std(daily_returns) * np.sqrt(252)
        
        if std_return == 0:
            return 0.0
        
        # Compute Sharpe Ratio
        sharpe = (mean_return - risk_free_rate) / std_return
        return sharpe
        
    except Exception as e:
        print(f"Sharpe Ratio error: {e}")
        return 0.0


def calculate_sortino_ratio(stock_data, lookback_days=90, risk_free_rate=0.04):
    """Compute Sortino Ratio for ranking."""
    try:
        if len(stock_data) < 10:
            return 0.0
        
        # Get close prices
        if 'close' in stock_data.columns:
            close_prices = stock_data['close'].values
        elif 'Close' in stock_data.columns:
            close_prices = stock_data['Close'].values
        else:
            return 0.0
        
        # Compute daily returns
        daily_returns = np.diff(close_prices) / close_prices[:-1]
        daily_returns = daily_returns[~np.isnan(daily_returns)]
        
        if len(daily_returns) < 5:
            return 0.0
        
        # Compute annualized return
        mean_return = np.mean(daily_returns) * 252
        
        # Compute downside volatility (negative returns only)
        negative_returns = daily_returns[daily_returns < 0]
        if len(negative_returns) == 0:
            return float('inf') if mean_return > risk_free_rate else 0.0
        
        downside_std = np.std(negative_returns) * np.sqrt(252)
        
        if downside_std == 0:
            return 0.0
        
        # Compute Sortino Ratio
        sortino = (mean_return - risk_free_rate) / downside_std
        return sortino
        
    except Exception as e:
        print(f"Sortino Ratio error: {e}")
        return 0.0


class RatioBasedSelector:
    """Stock selector based on risk-adjusted ratios."""
    
    def __init__(self, ratio_type='sharpe', lookback_days=90):
        """Initialize selector."""
        self.ratio_type = ratio_type
        self.lookback_days = lookback_days
    
    def rank_stocks(self, all_stock_data, current_date, n_top=3):
        """Rank stocks based on the selected ratio."""
        ratios = {}
        
        for symbol, data in all_stock_data.items():
            # Get lookback window data
            if 'date' in data.columns:
                data = data.copy()
                data['date'] = pd.to_datetime(data['date'])
                current_dt = pd.to_datetime(current_date)
                window_start = current_dt - pd.Timedelta(days=self.lookback_days)
                window_data = data[(data['date'] >= window_start) & (data['date'] <= current_dt)]
            else:
                window_data = data.tail(self.lookback_days)
            
            if len(window_data) < 10:
                continue
            
            if self.ratio_type == 'sharpe':
                ratio = calculate_sharpe_ratio(window_data, self.lookback_days)
            else:
                ratio = calculate_sortino_ratio(window_data, self.lookback_days)
            
            ratios[symbol] = ratio
        
        # Sort and return top-n
        sorted_stocks = sorted(ratios.items(), key=lambda x: x[1], reverse=True)
        return [s[0] for s in sorted_stocks[:n_top]]


class RatioSelectionEnvironment(TradingEnvironment):
    """Trading environment using risk-adjusted selection."""
    
    def __init__(self, config, data_loader, lim_calculator, ratio_type='sharpe'):
        super().__init__(config, data_loader, lim_calculator)
        self.ratio_selector = RatioBasedSelector(
            ratio_type=ratio_type,
            lookback_days=config['lim']['lookback_days']
        )
        self.ratio_type = ratio_type
    
    def _select_portfolio_stocks(self, group_type='TOP'):
        """Override stock selection using risk-adjusted ratios."""
        # Get current date
        current_date = self.market_data.iloc[self.current_step]['date']
        portfolio_size = self.config['portfolio']['size']
        
        # Select stocks using Sharpe/Sortino ratio
        selected = self.ratio_selector.rank_stocks(
            self.data_loader.all_stock_data,
            current_date,
            n_top=portfolio_size
        )
        
        if len(selected) < portfolio_size:
            # If insufficient, fill with remaining available stocks
            avaSIMble = list(self.data_loader.all_stock_data.keys())
            avaSIMble = [s for s in avaSIMble if s not in selected]
            selected.extend(avaSIMble[:portfolio_size - len(selected)])
        
        self.current_stocks = selected
        print(f"Selected stocks using {self.ratio_type.upper()} Ratio: {selected}")
        
        # Update current holdings data
        self.current_stock_data = {}
        for symbol in self.current_stocks:
            if symbol in self.data_loader.all_stock_data:
                self.current_stock_data[symbol] = self.data_loader.all_stock_data[symbol]
        
        # Initialize equal weights
        equal_weight = 1.0 / len(self.current_stocks) if self.current_stocks else 0
        self.current_weights = {symbol: equal_weight for symbol in self.current_stocks}


def run_ratio_selection_experiment(config, ratio_type, 
                                    train_start, train_end, test_start, test_end):
    """Run ablation using risk-adjusted ratios."""
    print(f"\n{'='*60}")
    print(f"Running {ratio_type.upper()} Ratio selection strategy...")
    print(f"{'='*60}")
    
    data_dir = os.path.join(os.getcwd(), 'data')
    
    # ========== Training phase ==========
    print(f"\n--- Training phase: {train_start} to {train_end} ---")
    
    # Create train data loader
    train_data_loader = DataLoader(
        start_date=train_start,
        end_date=train_end,
        symbols=config['data']['symbols'],
        data_dir=data_dir
    )
    train_data_loader.load_data()
    
    # Create LIM calculator
    lim_calculator = LIMCalculator(
        window_size=config['lim']['window_size'],
        alpha=config['lim']['alpha'],
        lookback_days=config['lim']['lookback_days']
    )
    
    # Update config for training
    train_config = config.copy()
    train_config['data'] = config['data'].copy()
    train_config['data']['start_date'] = train_start
    train_config['data']['end_date'] = train_end
    
    # Create training environment
    train_env = RatioSelectionEnvironment(
        config=train_config,
        data_loader=train_data_loader,
        lim_calculator=lim_calculator,
        ratio_type=ratio_type
    )
    
    # Create PPO model
    model = PPOModel(
        state_dim=train_env.observation_space.shape[0],
        action_dim=train_env.action_space.shape[0],
        lr=config['rl']['learning_rate'],
        gamma=config['rl']['gamma'],
        clip_ratio=config['rl']['clip_ratio'],
        value_coef=config['rl']['value_coef'],
        entropy_coef=config['rl']['entropy_coef']
    )
    
    # Train model
    model.train(train_env, epochs=config['rl']['epochs'])
    
    # ========== Testing phase ==========
    print(f"\n--- Testing phase: {test_start} to {test_end} ---")
    
    # Create test data loader
    test_data_loader = DataLoader(
        start_date=test_start,
        end_date=test_end,
        symbols=config['data']['symbols'],
        data_dir=data_dir
    )
    test_data_loader.load_data()
    
    # Update config for testing
    test_config = config.copy()
    test_config['data'] = config['data'].copy()
    test_config['data']['start_date'] = test_start
    test_config['data']['end_date'] = test_end
    
    # Create test environment
    test_env = RatioSelectionEnvironment(
        config=test_config,
        data_loader=test_data_loader,
        lim_calculator=lim_calculator,
        ratio_type=ratio_type
    )
    
    # Create portfolio instance
    portfolio = Portfolio(
        initial_capital=config['portfolio']['initial_capital'],
        commission_rate=config['trading']['commission_rate']
    )
    
    # Run strategy on test set
    test_env.reset(start_date=test_start, end_date=test_end)
    portfolio.run_strategy(model, test_env)
    
    # Collect results
    results = portfolio.get_performance_summary()
    results['dates'] = portfolio.history['date']
    results['portfolio_values'] = portfolio.history['portfolio_value']
    results['strategy_name'] = f'{ratio_type}_selection'
    
    return results


def main():
    """Main entry point."""
    print("="*60)
    print("Ablation study: Sharpe vs Sortino selection")
    print("="*60)
    
    # Training/testing periods from the paper
    TRAIN_START = "2019-06-12"
    TRAIN_END = "2024-04-05"
    TEST_START = "2024-04-06"
    TEST_END = "2025-04-06"
    
    # Load config
    config = load_config('config/default.yaml')
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config['paths']['results_path'], f'ratio_selection_ablation_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize data loader - full date range
    data_dir = os.path.join(os.getcwd(), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"Training: {TRAIN_START} to {TRAIN_END}")
    print(f"Testing: {TEST_START} to {TEST_END}")
    
    # Store all results
    all_results = []
    
    # Run Sharpe selection experiment
    sharpe_results = run_ratio_selection_experiment(
        config, 
        ratio_type='sharpe',
        train_start=TRAIN_START, train_end=TRAIN_END,
        test_start=TEST_START, test_end=TEST_END
    )
    all_results.append({
        'Strategy': 'sharpe_selection',
        'TR': sharpe_results.get('total_return', 0),
        'AR': sharpe_results.get('annual_return', 0),
        'SR': sharpe_results.get('sharpe_ratio', 0),
        'SoR': sharpe_results.get('sortino_ratio', 0),
        'VR': sharpe_results.get('volatility', 0),
        'MDD': sharpe_results.get('max_drawdown', 0),
        'Win_rate': sharpe_results.get('win_rate', 0),
        'PnL': sharpe_results.get('profit_loss_ratio', 0)
    })
    print(f"\nSharpe selection results: TR={sharpe_results.get('total_return', 0):.4f}, "
          f"AR={sharpe_results.get('annual_return', 0):.4f}, "
          f"SR={sharpe_results.get('sharpe_ratio', 0):.4f}")
    
    # Run Sortino selection experiment
    sortino_results = run_ratio_selection_experiment(
        config,
        ratio_type='sortino',
        train_start=TRAIN_START, train_end=TRAIN_END,
        test_start=TEST_START, test_end=TEST_END
    )
    all_results.append({
        'Strategy': 'sortino_selection',
        'TR': sortino_results.get('total_return', 0),
        'AR': sortino_results.get('annual_return', 0),
        'SR': sortino_results.get('sharpe_ratio', 0),
        'SoR': sortino_results.get('sortino_ratio', 0),
        'VR': sortino_results.get('volatility', 0),
        'MDD': sortino_results.get('max_drawdown', 0),
        'Win_rate': sortino_results.get('win_rate', 0),
        'PnL': sortino_results.get('profit_loss_ratio', 0)
    })
    print(f"\nSortino selection results: TR={sortino_results.get('total_return', 0):.4f}, "
          f"AR={sortino_results.get('annual_return', 0):.4f}, "
          f"SR={sortino_results.get('sharpe_ratio', 0):.4f}")
    
    # Save results
    results_df = pd.DataFrame(all_results)
    results_path = os.path.join(output_dir, 'ratio_selection_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")
    
    # Print summary table
    print("\n" + "="*60)
    print("Ablation study summary")
    print("="*60)
    print(results_df.to_string(index=False))
    
    return results_df


if __name__ == '__main__':
    main()


