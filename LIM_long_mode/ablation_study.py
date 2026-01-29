import os
import numpy as np
import pandas as pd
import yaml
from datetime import datetime
from utils.environment import TradingEnvironment
from models.ppo_agent import PPOAgent as PPOModel
from visualization.ablation_visualizer import AblationVisualizer
from utils.data_loader import DataLoader
from utils.lim_calculator import LIMCalculator
import argparse
import time
from utils.portfolio import Portfolio
import matplotlib.pyplot as plt


# ========== Sharpe/Sortino ratio helpers ==========

def calculate_sharpe_ratio(stock_data, lookback_days=90, risk_free_rate=0.04):
    """Compute Sharpe ratio for ranking."""
    try:
        if len(stock_data) < 10:
            return 0.0
        
        if 'close' in stock_data.columns:
            close_prices = stock_data['close'].values
        elif 'Close' in stock_data.columns:
            close_prices = stock_data['Close'].values
        else:
            return 0.0
        
        daily_returns = np.diff(close_prices) / close_prices[:-1]
        daily_returns = daily_returns[~np.isnan(daily_returns)]
        
        if len(daily_returns) < 5:
            return 0.0
        
        mean_return = np.mean(daily_returns) * 252
        std_return = np.std(daily_returns) * np.sqrt(252)
        
        if std_return == 0:
            return 0.0
        
        return (mean_return - risk_free_rate) / std_return
        
    except Exception:
        return 0.0


def calculate_sortino_ratio(stock_data, lookback_days=90, risk_free_rate=0.04):
    """Compute Sortino ratio for ranking."""
    try:
        if len(stock_data) < 10:
            return 0.0
        
        if 'close' in stock_data.columns:
            close_prices = stock_data['close'].values
        elif 'Close' in stock_data.columns:
            close_prices = stock_data['Close'].values
        else:
            return 0.0
        
        daily_returns = np.diff(close_prices) / close_prices[:-1]
        daily_returns = daily_returns[~np.isnan(daily_returns)]
        
        if len(daily_returns) < 5:
            return 0.0
        
        mean_return = np.mean(daily_returns) * 252
        negative_returns = daily_returns[daily_returns < 0]
        
        if len(negative_returns) == 0:
            return float('inf') if mean_return > risk_free_rate else 0.0
        
        downside_std = np.std(negative_returns) * np.sqrt(252)
        
        if downside_std == 0:
            return 0.0
        
        return (mean_return - risk_free_rate) / downside_std
        
    except Exception:
        return 0.0


class RatioBasedSelector:
    """Stock selector based on risk-adjusted metrics."""
    
    def __init__(self, ratio_type='sharpe', lookback_days=90):
        self.ratio_type = ratio_type
        self.lookback_days = lookback_days
    
    def rank_stocks(self, all_stock_data, current_date, n_top=3):
        """Rank stocks by the configured metric."""
        ratios = {}
        
        for symbol, data in all_stock_data.items():
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
        
        sorted_stocks = sorted(ratios.items(), key=lambda x: x[1], reverse=True)
        return [s[0] for s in sorted_stocks[:n_top]]


class RatioSelectionEnvironment(TradingEnvironment):
    """Trading environment using risk-adjusted metrics for selection."""
    
    def __init__(self, config, data_loader, lim_calculator, ratio_type='sharpe'):
        super().__init__(config, data_loader, lim_calculator)
        self.ratio_selector = RatioBasedSelector(
            ratio_type=ratio_type,
            lookback_days=config['lim']['lookback_days']
        )
        self.ratio_type = ratio_type
    
    def _select_portfolio_stocks(self, group_type='TOP'):
        """Override selection using risk-adjusted metrics."""
        current_date = self.market_data.iloc[self.current_step]['date']
        portfolio_size = self.config['portfolio']['size']
        
        selected = self.ratio_selector.rank_stocks(
            self.data_loader.all_stock_data,
            current_date,
            n_top=portfolio_size
        )
        
        if len(selected) < portfolio_size:
            avaSIMble = list(self.data_loader.all_stock_data.keys())
            avaSIMble = [s for s in avaSIMble if s not in selected]
            selected.extend(avaSIMble[:portfolio_size - len(selected)])
        
        self.current_stocks = selected
        print(f"Select stocks using {self.ratio_type.upper()} ratio: {selected}")
        
        self.current_stock_data = {}
        for symbol in self.current_stocks:
            if symbol in self.data_loader.all_stock_data:
                self.current_stock_data[symbol] = self.data_loader.all_stock_data[symbol]
        
        equal_weight = 1.0 / len(self.current_stocks) if self.current_stocks else 0
        self.current_weights = {symbol: equal_weight for symbol in self.current_stocks}

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Run ablation study')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                      help='Run mode: train or test')
    parser.add_argument('--config', type=str, default='config/default.yaml',
                      help='Config file path')
    parser.add_argument('--start_date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--group_type', type=str, default='TOP',
                      choices=['TOP', 'MIDDLE', 'LOW'],
                      help='Stock group type')
    return parser.parse_args()

def load_config(config_path):
    """Load configuration file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

class AblationStudy:
    """Ablation study runner."""
    
    def __init__(self, config, mode='train', start_date=None, end_date=None):
        """Initialize the ablation study.

        Args:
            config (dict): Configuration dictionary.
            mode (str): Run mode ('train' or 'test').
            start_date (str): Start date.
            end_date (str): End date.
        """
        self.config = config
        self.mode = mode
        
        # Update date range if provided.
        if start_date:
            self.config['data']['start_date'] = start_date
        if end_date:
            self.config['data']['end_date'] = end_date
            
        # Create output directory.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(config['paths']['results_path'], f'ablation_study_{timestamp}')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Ensure data directory exists.
        data_dir = os.path.join(os.getcwd(), 'data')
        os.makedirs(data_dir, exist_ok=True)
        
        print("Initializing data loader...")
        print(f"Date range: {self.config['data']['start_date']} to {self.config['data']['end_date']}")
        print(f"Symbols: {self.config['data']['symbols']}")
        
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
        
        # Initialize visualizer.
        self.visualizer = AblationVisualizer(self.output_dir)
        
        # Save experiment config.
        self._save_config()
    
    def _save_config(self):
        """Save experiment configuration."""
        config_path = os.path.join(self.output_dir, 'experiment_config.yaml')
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
    
    def run_baseline(self):
        """Run the baseline model."""
        print("Running baseline model...")
        env = TradingEnvironment(
            config=self.config,
            data_loader=self.data_loader,
            lim_calculator=self.lim_calculator
        )
        model = PPOModel(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            lr=self.config['rl']['learning_rate'],
            gamma=self.config['rl']['gamma'],
            clip_ratio=self.config['rl']['clip_ratio'],
            value_coef=self.config['rl']['value_coef'],
            entropy_coef=self.config['rl']['entropy_coef']
        )
        
        if self.mode == 'train':
            model.train(env, epochs=self.config['rl']['epochs'])
        else:
            model.load(self.config['paths']['model_load_path'])
        
        portfolio = Portfolio(
            initial_capital=self.config['portfolio']['initial_capital'],
            commission_rate=self.config['trading']['commission_rate']
        )
        
        env.reset()
        portfolio.run_strategy(model, env)
        
        results = portfolio.get_performance_summary()
        
        results['dates'] = portfolio.history['date']
        results['portfolio_values'] = portfolio.history['portfolio_value']
        
        return results
    
    def run_equal_weight(self):
        """Run equal-weight baseline strategy."""
        print("Running equal-weight strategy...")
        config = self.config.copy()
        config['portfolio']['equal_weight'] = True
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
        
        return results
    
    def run_fixed_15day(self):
        """Run fixed 15-day rebalance strategy."""
        print("Running fixed 15-day rebalance strategy...")
        config = self.config.copy()
        config['environment']['rebalance_period'] = 15
        config['environment']['dynamic_rebalance'] = False
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
        
        return results
    
    def run_fixed_30day(self):
        """Run fixed 30-day rebalance strategy."""
        print("Running fixed 30-day rebalance strategy...")
        config = self.config.copy()
        config['environment']['rebalance_period'] = 30
        config['environment']['dynamic_rebalance'] = False
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
        
        return results
    
    def run_fixed_60day(self):
        """Run fixed 60-day rebalance strategy."""
        print("Running fixed 60-day rebalance strategy...")
        config = self.config.copy()
        config['environment']['rebalance_period'] = 60
        config['environment']['dynamic_rebalance'] = False
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
        
        return results
    
    def run_fixed_90day(self):
        """Run fixed 90-day rebalance strategy."""
        print("Running fixed 90-day rebalance strategy...")
        config = self.config.copy()
        config['environment']['rebalance_period'] = 90
        config['environment']['dynamic_rebalance'] = False
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
        
        return results
    
    def run_random_rebalance(self):
        """Run random rebalance strategy."""
        print("Running random rebalance strategy...")
        config = self.config.copy()
        config['environment']['random_rebalance'] = True
        config['environment']['dynamic_rebalance'] = False
        config['environment']['random_rebalance_mean'] = 30  # Mean 30 days
        config['environment']['random_rebalance_var'] = 15   # Variance 15 days
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
        
        return results
    
    def run_literature_rl(self):
        """Run the literature RL baseline strategy."""
        print("Running literature RL baseline strategy...")
        config = self.config.copy()
        config['rl']['use_literature_baseline'] = True
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
            entropy_coef=config['rl']['entropy_coef'],
            use_literature_baseline=True
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
        
        return results
    
    def run_mean_var(self):
        """Run mean-variance optimization strategy."""
        print("Running mean-variance strategy...")
        config = self.config.copy()
        config['portfolio']['use_mean_var'] = True
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
        
        return results
    
    def run_momentum(self):
        """Run momentum strategy."""
        print("Running momentum strategy...")
        config = self.config.copy()
        config['portfolio']['use_momentum'] = True
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
        
        return results
    
    def run_no_LIM(self):
        """Run model without LIM* ranking."""
        print("Running model without LIM* ranking...")
        config = self.config.copy()
        config['lim']['use_LIM'] = False
        env = TradingEnvironment(
            config=config,
            data_loader=self.data_loader,
            lim_calculator=None  # Disable LIM calculator
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
        
        return results
    
    def run_sharpe_selection(self):
        """Run Sharpe-ratio stock selection strategy."""
        print("Running Sharpe ratio stock selection strategy...")
        config = self.config.copy()
        
        env = RatioSelectionEnvironment(
            config=config,
            data_loader=self.data_loader,
            lim_calculator=self.lim_calculator,
            ratio_type='sharpe'
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
        
        return results
    
    def run_sortino_selection(self):
        """Run Sortino-ratio stock selection strategy."""
        print("Running Sortino ratio stock selection strategy...")
        config = self.config.copy()
        
        env = RatioSelectionEnvironment(
            config=config,
            data_loader=self.data_loader,
            lim_calculator=self.lim_calculator,
            ratio_type='sortino'
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
        
        return results
    
    def run_all_experiments(self):
        """Run all ablation experiments."""
        print("Starting ablation experiments...")
        
        results = {
            'baseline': self.run_baseline()
        }
        
        if self.config['ablation'].get('run_equal_weight', False):
            results['equal_weight'] = self.run_equal_weight()
        
        if self.config['ablation'].get('run_fixed_15day', False):
            results['fixed_15day'] = self.run_fixed_15day()
            
        if self.config['ablation'].get('run_fixed_30day', False):
            results['fixed_30day'] = self.run_fixed_30day()
            
        if self.config['ablation'].get('run_fixed_60day', False):
            results['fixed_60day'] = self.run_fixed_60day()
            
        if self.config['ablation'].get('run_fixed_90day', False):
            results['fixed_90day'] = self.run_fixed_90day()
            
        if self.config['ablation'].get('run_random_rebalance', False):
            results['random_rebalance'] = self.run_random_rebalance()
            
        if self.config['ablation'].get('run_literature_rl', False):
            results['literature_rl'] = self.run_literature_rl()
            
        if self.config['ablation'].get('run_mean_var', False):
            results['mean_var'] = self.run_mean_var()
            
        if self.config['ablation'].get('run_momentum', False):
            results['momentum'] = self.run_momentum()
            
        if self.config['ablation'].get('run_no_LIM', False):
            results['no_LIM'] = self.run_no_LIM()
        
        if self.config['ablation'].get('run_sharpe_selection', False):
            results['sharpe_selection'] = self.run_sharpe_selection()
            
        if self.config['ablation'].get('run_sortino_selection', False):
            results['sortino_selection'] = self.run_sortino_selection()
        
        metrics_path, portfolio_path = self._save_results(results)
        
        self._visualize_results(results)
        
        if self.mode == 'test':
            print("\nStarting SHAP model interpretability analysis...")
            
            baseline_model = PPOModel(
                state_dim=self.config['rl']['state_dim'],
                action_dim=self.config['rl']['action_dim'],
                lr=self.config['rl']['learning_rate'],
                gamma=self.config['rl']['gamma'],
                clip_ratio=self.config['rl']['clip_ratio'],
                value_coef=self.config['rl']['value_coef'],
                entropy_coef=self.config['rl']['entropy_coef']
            )
            baseline_model.load(self.config['paths']['model_load_path'])
            
            baseline_shap = self.analyze_model_with_shap(baseline_model, "baseline")
            
            shap_results = {
                'baseline': baseline_shap
            }
            
            if self.config['ablation'].get('run_no_LIM', False):
                no_LIM_model = PPOModel(
                    state_dim=self.config['rl']['state_dim'],
                    action_dim=self.config['rl']['action_dim'],
                    lr=self.config['rl']['learning_rate'],
                    gamma=self.config['rl']['gamma'],
                    clip_ratio=self.config['rl']['clip_ratio'],
                    value_coef=self.config['rl']['value_coef'],
                    entropy_coef=self.config['rl']['entropy_coef']
                )
                model_path = os.path.join(self.config['paths']['model_save_path'], 'no_LIM_model')
                if os.path.exists(model_path):
                    no_LIM_model.load(model_path)
                    shap_results['no_LIM'] = self.analyze_model_with_shap(no_LIM_model, "no_LIM")
            
            results['shap_analysis'] = shap_results
        
        return results, metrics_path, portfolio_path
    
    def _save_results(self, results):
        """Save experiment results.

        Args:
            results (dict): Experiment results.
        """
        metrics_data = []
        for strategy, data in results.items():
            metrics = {
                'strategy': strategy,
                'total_return': data.get('total_return', 0),
                'sharpe_ratio': data.get('sharpe_ratio', 0),
                'max_drawdown': data.get('max_drawdown', 0),
                'win_rate': data.get('win_rate', 0),
                'volatility': data.get('volatility', 0),
                'sortino_ratio': data.get('sortino_ratio', 0),
                'turnover_rate': data.get('turnover_rate', 0),
                'stop_loss_count': data.get('stop_loss_count', 0)
            }
            metrics_data.append(metrics)
        
        df = pd.DataFrame(metrics_data)
        metrics_path = os.path.join(self.output_dir, f'ablation_metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        df.to_csv(metrics_path, index=False)
        print(f"Metrics saved to: {metrics_path}")
        
        portfolio_data = {}
        for strategy, data in results.items():
            if 'dates' in data and 'portfolio_values' in data:
                portfolio_data[strategy] = pd.DataFrame({
                    'date': data['dates'],
                    'portfolio_value': data['portfolio_values']
                })
        
        if portfolio_data:
            all_data = pd.DataFrame()
            for strategy, df in portfolio_data.items():
                all_data[f'{strategy}_date'] = df['date']
                all_data[f'{strategy}_value'] = df['portfolio_value']
            
            portfolio_path = os.path.join(self.output_dir, f'portfolio_values_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
            all_data.to_csv(portfolio_path, index=False)
            print(f"Portfolio value data saved to: {portfolio_path}")
        else:
            print("Warning: no portfolio value data available")
            portfolio_path = None
        
        return metrics_path, portfolio_path
    
    def _visualize_results(self, results):
        """Visualize experiment results.

        Args:
            results (dict): Experiment results.
        """
        plt.rcParams['font.family'] = 'Times New Roman'
        
        portfolio_data = {}
        for strategy, data in results.items():
            if 'dates' in data and 'portfolio_values' in data:
                portfolio_data[strategy] = {
                    'date': data['dates'],
                    'portfolio_value': data['portfolio_values']
                }
        
        portfolio_plot_path = self.visualizer.plot_ablation_portfolio_values(
            portfolio_data,
            title='Portfolio Value Comparison (Ablation Study)'
        )
        print(f"Portfolio comparison plot saved to: {portfolio_plot_path}")
        
        metrics_plot_path = self.visualizer.plot_ablation_metrics(
            results,
            metrics=['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate', 
                    'volatility', 'sortino_ratio', 'turnover_rate', 'stop_loss_count']
        )
        print(f"Metrics comparison plot saved to: {metrics_plot_path}")

    def analyze_model_with_shap(self, model, model_name="baseline"):
        """Analyze model with SHAP for interpretability.

        Args:
            model: Trained model.
            model_name: Model name for output files.

        Returns:
            dict: SHAP analysis results.
        """
        import shap
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        
        print(f"Running SHAP analysis for model: {model_name}...")
        
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = 'Times New Roman'
        plt.rcParams['font.size'] = 13
        
        shap_dir = os.path.join(self.output_dir, 'shap_analysis')
        os.makedirs(shap_dir, exist_ok=True)
        
        try:
            env = TradingEnvironment(
                config=self.config,
                data_loader=self.data_loader,
                lim_calculator=self.lim_calculator
            )
            
            print("Collecting sample data...")
            state = env.reset()
            states = []
            
            sample_count = 0
            done = False
            
            while not done and sample_count < 200:
                action = model.predict(state)
                states.append(state)
                next_state, reward, done, _ = env.step(action)
                state = next_state
                sample_count += 1
            
            print(f"Collected {len(states)} state samples")
            
            states_array = np.array(states)
            
            feature_names = []
            
            feature_names.extend(["Market Return", "Market Volatility", "Market Momentum"])
            
            for i in range(env.portfolio_size):
                stock_features = [f"Stock {i+1} Return", f"Stock {i+1} Volatility"]
                feature_names.extend(stock_features)
            
            feature_names.extend([
                "Portfolio Value Ratio",
                "Max Drawdown",
                "Return Change",
                "Cash Ratio",
                "Stop-Loss Status"
            ])
            
            if len(feature_names) < env.observation_space.shape[0]:
                for i in range(len(feature_names), env.observation_space.shape[0]):
                    feature_names.append(f"Feature {i+1}")
            
            if len(feature_names) > env.observation_space.shape[0]:
                feature_names = feature_names[:env.observation_space.shape[0]]
            
            X_train = pd.DataFrame(states_array, columns=feature_names)
            
            if hasattr(model, 'get_booster') or hasattr(model, 'booster_'):
                print("Using TreeExplainer...")
                explainer = shap.TreeExplainer(model)
            else:
                print("Using KernelExplainer (for deep RL models)...")
                background = X_train.iloc[:50]  # Use 50 samples as background
                
                def model_predict(x):
                    return np.array([model.predict(x_i) for x_i in x])
                
                explainer = shap.KernelExplainer(model_predict, background)
            
            print("Computing SHAP values (may take a few minutes)...")
            shap_values = explainer.shap_values(X_train)
            
            plt.figure(figsize=(12, 10))
            shap.summary_plot(
                shap_values, 
                X_train, 
                feature_names=feature_names, 
                plot_type="dot",
                cmap="RdBu",
                show=False
            )
            plt.tight_layout()
            summary_path = os.path.join(shap_dir, f'{model_name}_shap_summary.png')
            plt.savefig(summary_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"SHAP summary plot saved to: {summary_path}")
            
            try:
                print("Computing SHAP interaction values...")
                sample_size = min(100, len(X_train))
                X_sample = X_train.iloc[:sample_size]
                
                shap_interaction_values = explainer.shap_interaction_values(X_sample)
                
                plt.figure(figsize=(14, 12))
                shap.summary_plot(
                    shap_interaction_values, 
                    X_sample,
                    max_display=15,
                    show=False
                )
                plt.tight_layout()
                interaction_path = os.path.join(shap_dir, f'{model_name}_interaction_summary.png')
                plt.savefig(interaction_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"SHAP interaction summary plot saved to: {interaction_path}")
                
                plt.figure(figsize=(12, 8))
                shap.summary_plot(
                    shap_values, 
                    X_train, 
                    plot_type="bar",
                    feature_names=feature_names,
                    max_display=20,
                    show=False
                )
                bar_path = os.path.join(shap_dir, f'{model_name}_feature_importance.png')
                plt.savefig(bar_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"Feature importance bar chart saved to: {bar_path}")
                
            except Exception as e:
                print(f"Error computing SHAP interaction values: {str(e)}")
                print("This may be due to memory limits; skipping interaction analysis.")
            
            return {
                'shap_values': shap_values,
                'feature_names': feature_names,
                'model_name': model_name
            }
            
        except Exception as e:
            print(f"Error during SHAP analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            return {}

if __name__ == '__main__':
    args = parse_args()
    
    config = load_config(args.config)
    
    study = AblationStudy(
        config=config,
        mode=args.mode,
        start_date=args.start_date,
        end_date=args.end_date
    )
    
    config['environment']['group_type'] = args.group_type
    
    results, metrics_path, portfolio_path = study.run_all_experiments() 

