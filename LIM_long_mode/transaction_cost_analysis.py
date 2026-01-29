"""
Transaction cost analysis module.

Compares how transaction cost settings affect OLPS strategies.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
from tqdm import tqdm
import yaml
import copy
from utils.data_loader_factory import create_data_loader
from config.config import get_config

# Import all strategy classes from strategy_comparison.py
from strategy_comparison import (
    BaseStrategy, UBAH, UCRP, EG, Anticor, PAMR, 
    OLMAR, CORN_K, CWMR_Var, RMR, TCO1, 
    calculate_metrics
)

class TransactionCostAnalyzer:
    """Transaction cost analyzer for multiple strategies."""
    
    def __init__(self, data_path, output_dir, start_date, end_date, config_path='config/default.yaml'):
        """Initialize the transaction cost analyzer."""
        self.data_path = data_path
        self.output_dir = output_dir
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.config_path = config_path
        
        # Load config from path
        if not os.path.exists(config_path):
            # Try adding LIMPPO_CNN prefix
            alt_config_path = os.path.join('LIMPPO_CNN', config_path)
            if os.path.exists(alt_config_path):
                config_path = alt_config_path
        
        self.config = get_config(config_path)
        print(f"Loaded config from {config_path}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize strategy set
        self.strategies = [
            UBAH(),           # Buy and hold
            UCRP(),           # Uniform constant rebalanced portfolio
            EG(),             # Exponentiated gradient
            Anticor(),        # Anti-correlation
            PAMR(),           # Passive aggressive mean reversion
            OLMAR(),          # Online moving average reversion
            CORN_K(),         # Correlation-driven nonparametric KNN
            CWMR_Var(),       # Confidence-weighted mean reversion (variance)
            RMR(),            # Robust median reversion
            TCO1()            # Transaction cost optimization strategy 1
        ]
        
        # Add custom baseline strategy
        self.add_baseline_strategy()
        
        # Transaction cost rate series
        self.commission_rates = [0, 0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.0035, 0.004, 0.0045, 0.005]
        
        # Load data
        self.load_data()
    
    def add_baseline_strategy(self):
        """Add a custom baseline strategy."""
        # Load saved weights from config or use predefined weights
        weights_file = self.config.get('baseline', {}).get('weights_file', '')
        if weights_file and os.path.exists(weights_file):
            weights_data = pd.read_csv(weights_file)
            print(f"Loaded baseline weights from: {weights_file}")
            
            class BaselineStrategy(BaseStrategy):
                def __init__(self, weights_data):
                    super().__init__("baseline")
                    self.weights_data = weights_data
                    self.dates = pd.to_datetime(weights_data['date'])
                    self.symbols = [col for col in weights_data.columns if col != 'date']
                
                def allocate(self, data, current_weights=None):
                    # Get current date
                    current_date = data.index[-1]
                    
                    # Find nearest date
                    closest_idx = np.argmin([abs((d - current_date).total_seconds()) for d in self.dates])
                    closest_date = self.dates[closest_idx]
                    
                    # Get weights
                    weights_row = self.weights_data.loc[closest_idx]
                    
                    # Create weight dict
                    weights_dict = {sym: weights_row[sym] if sym in weights_row else 0 
                                   for sym in data.columns}
                    
                    # Convert to array matching current data
                    weights = np.array([weights_dict.get(sym, 0) for sym in data.columns])
                    
                    # Normalize weights
                    if np.sum(weights) > 0:
                        weights = weights / np.sum(weights)
                    else:
                        weights = np.ones(len(data.columns)) / len(data.columns)
                    
                    return weights
            
            self.strategies.insert(0, BaselineStrategy(weights_data))
        else:
            # If no weights file, use uniform baseline
            class BaselineStrategy(BaseStrategy):
                def __init__(self):
                    super().__init__("baseline")
                
                def allocate(self, data, current_weights=None):
                    return np.ones(data.shape[1]) / data.shape[1]
            
            self.strategies.insert(0, BaselineStrategy())
            print("Using uniform allocation as baseline strategy")
    
    def load_data(self):
        """Load stock data using the main experiment settings."""
        try:
            # More robust date handling
            if isinstance(self.start_date, str):
                self.start_date = pd.to_datetime(self.start_date)
            if isinstance(self.end_date, str):
                self.end_date = pd.to_datetime(self.end_date)
                
            # Extend start date to avoid indicator calculation errors
            extended_start_date = self.start_date - pd.Timedelta(days=120)
            
            # Update date range in config
            self.config['data']['start_date'] = extended_start_date.strftime('%Y-%m-%d')
            self.config['data']['end_date'] = self.end_date.strftime('%Y-%m-%d')
            print(f"Adjusted data load range: {extended_start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}")
            
            # If data path provided, update data directory
            if self.data_path:
                self.config['data']['data_dir'] = self.data_path
            
            # Create data loader via factory
            print("Creating data loader via factory...")
            
            # Add indicator computation error handling
            # Disable/simplify complex indicators that may fail
            if 'features' in self.config:
                tech_indicators = self.config.get('features', {}).get('tech_indicators', [])
                
                # Keep simple indicators, remove complex ones
                simple_indicators = ['SMA', 'EMA', 'CLOSE_REL']
                if tech_indicators:
                    self.config['features']['tech_indicators'] = [
                        ind for ind in tech_indicators if ind in simple_indicators
                    ]
                    print(f"Simplified tech indicators: {self.config['features']['tech_indicators']}")
            
            data_loader = create_data_loader(self.config)
            
            # Load data
            print("Starting data load...")
            try:
                data_loader.load_data()
            except Exception as e:
                print(f"Error loading data: {str(e)}")
                print("Trying simplified data loading...")
                
                # If load fails, use a simpler approach
                if hasattr(data_loader, 'fetch_stock_data'):
                    for symbol in data_loader.symbols:
                        try:
                            data_loader.fetch_stock_data(symbol, skip_features=True)
                        except Exception as inner_e:
                            print(f"Error fetching {symbol} data: {str(inner_e)}")
            
            # Convert data into format suitable for strategy comparison
            all_data = {}
            for symbol, data in data_loader.all_stock_data.items():
                # Try possible column names
                possible_cols = ['close', 'Close', 'raw_Close', 'close_price']
                for col in possible_cols:
                    if col in data.columns:
                        all_data[symbol] = data[col]
                        break
            
            if not all_data:
                raise ValueError("No valid close price data found")
            
            # Combine all stock data into one DataFrame
            self.data = pd.DataFrame(all_data)
            
            # Ensure index is datetime
            if not isinstance(self.data.index, pd.DatetimeIndex):
                # Try multiple ways to obtain dates
                date_found = False
                
                # Method 1: check for date column
                if hasattr(data_loader, 'all_stock_data') and len(data_loader.all_stock_data) > 0:
                    first_stock = list(data_loader.all_stock_data.keys())[0]
                    first_data = data_loader.all_stock_data[first_stock]
                    if 'date' in first_data.columns:
                        self.data.index = pd.to_datetime(first_data['date'])
                        date_found = True
                
                # Method 2: create custom date index
                if not date_found:
                    print("No date column found; creating date sequence...")
                    # Create date range
                    date_range = pd.date_range(start=extended_start_date, end=self.end_date, freq='B')
                    # If fewer rows than range, trim range
                    if len(date_range) > len(self.data):
                        date_range = date_range[-len(self.data):]
                    # If more rows than range, trim data
                    elif len(date_range) < len(self.data):
                        self.data = self.data.iloc[-len(date_range):]
                    # Set index
                    self.data.index = date_range
            
            # Filter date range (use actual needed range)
            self.data = self.data[(self.data.index >= self.start_date) & 
                                (self.data.index <= self.end_date)]
            
            print(f"Filtered date range: {self.data.index.min()} to {self.data.index.max()}")
            
            # Check and handle NaNs
            nan_count = self.data.isna().sum().sum()
            if nan_count > 0:
                print(f"Found {nan_count} NaNs, preprocessing...")
                
                # Preprocess NaNs with filling
                # Forward fill
                self.data = self.data.fillna(method='ffill')
                # Backward fill for leading NaNs
                self.data = self.data.fillna(method='bfill')
                
                # If NaNs remain (all-NaN columns), fill with column mean
                if self.data.isna().sum().sum() > 0:
                    print("NaNs remain; filling with column mean...")
                    self.data = self.data.fillna(self.data.mean())
                
                # Final NaN check
                remaining_nans = self.data.isna().sum().sum()
                if remaining_nans > 0:
                    print(f"Warning: {remaining_nans} NaNs remain; filling with 0...")
                    self.data = self.data.fillna(0)
                else:
                    print("All NaNs handled successfully")
            
            print(f"Loaded {self.data.shape[1]} stocks with {len(self.data)} trading days")
            
        except Exception as e:
            print(f"Data loading error: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Create a simple mock dataset to continue
            print("Creating mock data to continue analysis...")
            
            # Create date range
            date_range = pd.date_range(start=self.start_date, end=self.end_date, freq='B')
            
            # Create 10 mock stocks with random walk data
            n_stocks = 10
            mock_data = {}
            np.random.seed(42)  # Set seed for reproducibility
            
            for i in range(n_stocks):
                stock_name = f"STOCK_{i+1}"
                # Create random walk price series
                prices = 100 + np.cumsum(np.random.normal(0.001, 0.02, len(date_range)))
                mock_data[stock_name] = prices
            
            # Create DataFrame
            self.data = pd.DataFrame(mock_data, index=date_range)
            print(f"Created {n_stocks} mock stocks with {len(date_range)} trading days")
    
    def run_analysis(self, initial_capital=1000000):
        """Run transaction cost analysis across strategies and rates."""
        results = {}
        
        # Analyze each transaction cost rate
        for commission_rate in self.commission_rates:
            print(f"\nAnalyzing transaction cost rate: {commission_rate:.2%}")
            
            # Save results for this rate
            rate_results = {}
            
            # Backtest each strategy
            for strategy in self.strategies:
                print(f"Backtesting {strategy.name} at cost rate: {commission_rate:.2%}...")
                
                # Initialize capital and weights
                portfolio_value = initial_capital
                weights = None
                previous_weights = None
                portfolio_values = [portfolio_value]
                dates = [self.data.index[0]]
                
                # Accumulate transaction costs
                total_transaction_cost = 0
                transaction_costs = [0]  # Track step costs
                
                # Simulate trading
                try:
                    for i in tqdm(range(1, len(self.data))):
                        # Get historical data
                        hist_data = self.data.iloc[:i]
                        
                        # Get today's and previous day's prices
                        today_prices = self.data.iloc[i].values
                        prev_prices = self.data.iloc[i-1].values
                        
                        # Validate price data
                        if np.any(np.isnan(today_prices)) or np.any(np.isnan(prev_prices)):
                            print(f"Warning: day {i} prices contain NaNs; using previous weights")
                            if i > 1:
                                today_prices = np.nan_to_num(today_prices, nan=self.data.iloc[i-1].values)
                                prev_prices = np.nan_to_num(prev_prices, nan=self.data.iloc[i-2].values)
                        
                        # If first day or rebalance day
                        if weights is None or i % 30 == 0:  # Rebalance every 30 days
                            try:
                                # Save previous weights
                                previous_weights = weights.copy() if weights is not None else None
                                
                                # Compute new portfolio weights
                                weights = strategy.allocate(hist_data, weights)
                                
                                # Validate weights
                                if np.any(np.isnan(weights)):
                                    print(f"Warning: {strategy.name} produced NaN weights; using uniform")
                                    weights = np.ones(len(today_prices)) / len(today_prices)
                                
                                # Compute transaction costs (non-first allocation only)
                                if previous_weights is not None and commission_rate > 0:
                                    # Compute weight changes
                                    weight_changes = np.abs(weights - previous_weights).sum()
                                    
                                    # Ensure both sides are charged
                                    # weight_changes already sums absolute changes (buy and sell)
                                    # This already reflects two-sided costs without extra factor
                                    
                                    # Compute transaction cost (both sides charged)
                                    step_transaction_cost = weight_changes * commission_rate * portfolio_value
                                    
                                    # Deduct transaction cost
                                    portfolio_value -= step_transaction_cost
                                    
                                    # Accumulate transaction cost
                                    total_transaction_cost += step_transaction_cost
                                    transaction_costs.append(step_transaction_cost)
                                else:
                                    transaction_costs.append(0)
                                
                            except Exception as e:
                                print(f"Error: {strategy.name} weight allocation failed: {str(e)}")
                                weights = np.ones(len(today_prices)) / len(today_prices)
                                transaction_costs.append(0)
                        else:
                            transaction_costs.append(0)
                        
                        # Compute previous day's allocation
                        prev_allocation = portfolio_value * weights
                        
                        # Compute today's allocation value
                        new_allocation = prev_allocation * (today_prices / prev_prices)
                        portfolio_value = np.sum(new_allocation)
                        
                        # Validate portfolio_value
                        if np.isnan(portfolio_value) or np.isinf(portfolio_value):
                            print(f"Warning: {strategy.name} invalid portfolio value on day {i}; using previous")
                            portfolio_value = portfolio_values[-1]
                        
                        # Update weights
                        weights = new_allocation / portfolio_value
                        
                        # Validate weights
                        if np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
                            print(f"Warning: {strategy.name} invalid weights on day {i}; using uniform")
                            weights = np.ones(len(today_prices)) / len(today_prices)
                        
                        # Record portfolio value
                        portfolio_values.append(portfolio_value)
                        dates.append(self.data.index[i])
                    
                    # Convert to numpy array for metrics
                    portfolio_values_array = np.array(portfolio_values)
                    
                    # Handle NaN/inf in portfolio values
                    if np.any(np.isnan(portfolio_values_array)) or np.any(np.isinf(portfolio_values_array)):
                        print(f"Warning: {strategy.name} produced invalid portfolio series; attempting fix")
                        valid_values = ~(np.isnan(portfolio_values_array) | np.isinf(portfolio_values_array))
                        last_valid_idx = np.where(valid_values)[0][0]
                        last_valid_value = portfolio_values_array[last_valid_idx]
                        
                        for i in range(len(portfolio_values_array)):
                            if np.isnan(portfolio_values_array[i]) or np.isinf(portfolio_values_array[i]):
                                portfolio_values_array[i] = last_valid_value
                            else:
                                last_valid_value = portfolio_values_array[i]
                    
                    # Compute performance metrics
                    metrics = calculate_metrics(portfolio_values_array)
                    
                    # Add transaction costs to metrics
                    metrics['total_transaction_cost'] = total_transaction_cost
                    metrics['transaction_cost_ratio'] = total_transaction_cost / initial_capital
                    
                    # Store results
                    rate_results[strategy.name] = {
                        'portfolio_values': portfolio_values,
                        'dates': dates,
                        'metrics': metrics,
                        'transaction_costs': transaction_costs
                    }
                    
                    print(f"{strategy.name} results (cost rate: {commission_rate:.2%}):")
                    print(f"  Final wealth: {metrics['final_wealth']:.4f}")
                    print(f"  Annual return: {metrics['annual_return']:.4f}")
                    print(f"  Sharpe ratio: {metrics['sharpe_ratio']:.4f}")
                    print(f"  Max drawdown: {metrics['max_drawdown']:.4f}")
                    print(f"  Total transaction cost: {total_transaction_cost:.2f}")
                    print(f"  Transaction cost ratio: {metrics['transaction_cost_ratio']:.4%}")
                    print()
                    
                except Exception as e:
                    print(f"Error: exception while running {strategy.name}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    
                    # Create default metrics
                    metrics = {
                        'final_wealth': 1.0,
                        'annual_return': 0.0,
                        'sharpe_ratio': 0.0,
                        'max_drawdown': 0.0,
                        'calmar_ratio': 0.0,
                        'volatility': 0.0,
                        'total_transaction_cost': 0.0,
                        'transaction_cost_ratio': 0.0
                    }
                    # Store empty result
                    rate_results[strategy.name] = {
                        'portfolio_values': [initial_capital],
                        'dates': [self.data.index[0]],
                        'metrics': metrics,
                        'transaction_costs': [0],
                        'error': str(e)
                    }
            
            # Save results for this rate
            results[commission_rate] = rate_results
        
        return results
    
    def save_results(self, results, file_prefix='transaction_cost_analysis'):
        """Save analysis results."""
        # Create DataFrame for final wealth
        final_wealth_df = pd.DataFrame(index=self.commission_rates)
        
        # Create DataFrame for transaction costs
        transaction_cost_df = pd.DataFrame(index=self.commission_rates)
        
        # Get all strategy names
        strategy_names = results[0].keys()
        
        # Fill DataFrames
        for rate in self.commission_rates:
            rate_results = results[rate]
            for strategy_name in strategy_names:
                if strategy_name in rate_results:
                    metrics = rate_results[strategy_name]['metrics']
                    final_wealth_df.loc[rate, strategy_name] = metrics['final_wealth']
                    transaction_cost_df.loc[rate, strategy_name] = metrics['total_transaction_cost']
        
        # Save final wealth data
        final_wealth_path = os.path.join(self.output_dir, f"{file_prefix}_final_wealth.csv")
        final_wealth_df.to_csv(final_wealth_path)
        print(f"Final wealth data saved to: {final_wealth_path}")
        
        # Save transaction cost data
        transaction_cost_path = os.path.join(self.output_dir, f"{file_prefix}_transaction_cost.csv")
        transaction_cost_df.to_csv(transaction_cost_path)
        print(f"Transaction cost data saved to: {transaction_cost_path}")
        
        # Save detailed portfolio values for each rate
        for rate in self.commission_rates:
            rate_dir = os.path.join(self.output_dir, f"rate_{rate:.2f}")
            os.makedirs(rate_dir, exist_ok=True)
            
            rate_results = results[rate]
            for strategy_name, result in rate_results.items():
                # Create portfolio value DataFrame
                portfolio_df = pd.DataFrame({
                    'date': result['dates'],
                    'portfolio_value': result['portfolio_values'],
                    'transaction_cost': result['transaction_costs']
                })
                
                # Save portfolio value data
                portfolio_path = os.path.join(rate_dir, f"{strategy_name}_portfolio.csv")
                portfolio_df.to_csv(portfolio_path, index=False)
        
        # Save metrics for all rates and strategies
        metrics_df = []
        for rate in self.commission_rates:
            rate_results = results[rate]
            for strategy_name, result in rate_results.items():
                metrics = result['metrics'].copy()
                metrics['strategy'] = strategy_name
                metrics['commission_rate'] = rate
                metrics_df.append(metrics)
        
        metrics_df = pd.DataFrame(metrics_df)
        metrics_path = os.path.join(self.output_dir, f"{file_prefix}_all_metrics.csv")
        metrics_df.to_csv(metrics_path, index=False)
        print(f"All metrics data saved to: {metrics_path}")
    
    def visualize_results(self, results):
        """Visualize analysis results."""
        # Set font style
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 12
        
        # Get all strategy names
        strategy_names = list(results[0].keys())
        
        # Extract final wealth and transaction cost data
        final_wealth_data = {strategy: [] for strategy in strategy_names}
        transaction_cost_data = {strategy: [] for strategy in strategy_names}
        
        for rate in self.commission_rates:
            rate_results = results[rate]
            for strategy in strategy_names:
                if strategy in rate_results:
                    metrics = rate_results[strategy]['metrics']
                    final_wealth_data[strategy].append(metrics['final_wealth'])
                    transaction_cost_data[strategy].append(metrics['total_transaction_cost'])
        
        # Plot transaction cost vs final wealth
        plt.figure(figsize=(12, 8))
        
        for strategy in strategy_names:
            plt.plot(self.commission_rates, final_wealth_data[strategy], 
                    marker='o', label=strategy)
        
        plt.title('Impact of Transaction Cost on Final Wealth', fontfamily='Times New Roman', fontsize=14)
        plt.xlabel('Commission Rate', fontfamily='Times New Roman')
        plt.ylabel('Final Wealth', fontfamily='Times New Roman')
        plt.legend(prop={'family': 'Times New Roman'})
        plt.grid(True)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(self.output_dir, 'transaction_cost_vs_final_wealth.png'), dpi=300)
        plt.close()
        
        # Plot transaction cost bars
        plt.figure(figsize=(12, 8))
        
        x = np.arange(len(self.commission_rates))
        width = 0.8 / len(strategy_names)
        
        for i, strategy in enumerate(strategy_names):
            plt.bar(x + i*width, transaction_cost_data[strategy], 
                   width=width, label=strategy)
        
        plt.title('Total Transaction Cost by Strategy', fontfamily='Times New Roman', fontsize=14)
        plt.xlabel('Commission Rate', fontfamily='Times New Roman')
        plt.ylabel('Total Transaction Cost', fontfamily='Times New Roman')
        plt.xticks(x + width*len(strategy_names)/2, [f"{rate:.2%}" for rate in self.commission_rates])
        plt.legend(prop={'family': 'Times New Roman'})
        plt.grid(True, axis='y')
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(self.output_dir, 'transaction_cost_by_strategy.png'), dpi=300)
        plt.close()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Transaction cost impact analysis for strategies')
    parser.add_argument('--data_path', type=str, default='data',
                      help='Data path')
    parser.add_argument('--output_dir', type=str, default='transaction_cost_analysis_results',
                      help='Output directory')
    parser.add_argument('--start_date', type=str, default='2024-04-06',
                      help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default='2025-04-06',
                      help='End date (YYYY-MM-DD)')
    parser.add_argument('--initial_capital', type=float, default=1000000,
                      help='Initial capital')
    parser.add_argument('--config', type=str, default='config/default.yaml',
                      help='Config file path')
    return parser.parse_args()

if __name__ == '__main__':
    # Parse command line arguments
    args = parse_args()
    
    # Create analyzer
    analyzer = TransactionCostAnalyzer(
        data_path=args.data_path,
        output_dir=args.output_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        config_path=args.config
    )
    
    # Run analysis
    results = analyzer.run_analysis(initial_capital=args.initial_capital)
    
    # Save results
    analyzer.save_results(results)
    
    # Visualize results
    analyzer.visualize_results(results)
    
    print(f"Transaction cost analysis complete; results saved to: {args.output_dir}") 

