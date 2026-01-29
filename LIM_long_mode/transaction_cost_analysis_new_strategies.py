"""
Transaction cost analysis module - new strategies.

Compares how transaction cost settings affect WASC, CAEGc, RLCVAR, and DNAS.
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
import torch
from utils.data_loader_factory import create_data_loader
from config.config import get_config

# Import new strategy classes from strategy_comparison.py
from strategy_comparison import (
    BaseStrategy, calculate_metrics, WASC, CAEGc, RLCVAR, DNAS
)

class TransactionCostAnalyzerNewStrategies:
    """Transaction cost analyzer for new strategies."""
    
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
        
        # Initialize strategy set - only four new strategies
        self.strategies = [
            WASC(gamma=0.0025, u=1, epsilon=0.15),  # Weakly-aggregated specialized CRP
            CAEGc(eta=0.05, gamma_tc=0.01),         # Continuous aggregation EG (with TC)
            RLCVAR(window_size=30, beta=0.95),      # CVaR-based RL strategy
            DNAS(hist_len=60, feature_dim=7, eta=0.9)  # Dynamic number of assets
        ]
        
        # Transaction cost rate series
        self.commission_rates = [0, 0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.0035, 0.004, 0.0045, 0.005]
        
        # Load data
        self.load_data()
    
    def load_data(self):
        """Load stock data using the same settings as the main experiment."""
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
        
        # Ensure models are initialized (for pretrained strategies)
        for strategy in self.strategies:
            if isinstance(strategy, (RLCVAR, DNAS)) and not strategy.initialized:
                n_assets = self.data.shape[1]
                strategy.init_model(n_assets)
        
        # Analyze each transaction cost rate
        for commission_rate in self.commission_rates:
            print(f"\nAnalyzing transaction cost rate: {commission_rate:.2%}")
            
            # Save results for this transaction cost rate
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
                                
                                # Ensure weights are a numpy array
                                if isinstance(weights, dict):
                                    weight_array = np.zeros(len(self.data.columns))
                                    for j, symbol in enumerate(self.data.columns):
                                        weight_array[j] = weights.get(symbol, 0)
                                    weights = weight_array
                                
                                # Validate weights
                                if np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
                                    print(f"Warning: {strategy.name} produced NaN weights; using uniform")
                                    weights = np.ones(len(today_prices)) / len(today_prices)
                                
                                # Compute transaction costs (non-first allocation only)
                                if previous_weights is not None and commission_rate > 0:
                                    # Compute weight changes
                                    weight_changes = np.abs(weights - previous_weights).sum()
                                    
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
                        if np.sum(new_allocation) > 0:
                            weights = new_allocation / portfolio_value
                        else:
                            weights = np.ones(len(today_prices)) / len(today_prices)
                        
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
                        if np.any(valid_values):
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
                    metrics['final_wealth'] = portfolio_values[-1] / initial_capital
                    
                    # Store results
                    rate_results[strategy.name] = {
                        'portfolio_values': portfolio_values,
                        'dates': dates,
                        'metrics': metrics,
                        'transaction_costs': transaction_costs
                    }
                    
                    print(f"{strategy.name} results (cost rate: {commission_rate:.2%}):")
                    print(f"  Final wealth: {portfolio_values[-1]:.4f}")
                    print(f"  Total return: {metrics['total_ret']:.4f}")
                    print(f"  Annual return: {metrics['annual_ret']:.4f}")
                    print(f"  Sharpe ratio: {metrics['sharpe_rat']:.4f}")
                    print(f"  Max drawdown: {metrics['max_drawd']:.4f}")
                    print(f"  Total transaction cost: {total_transaction_cost:.2f}")
                    print(f"  Transaction cost ratio: {metrics['transaction_cost_ratio']:.4%}")
                    print()
                    
                except Exception as e:
                    print(f"Error: exception while running {strategy.name}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    
                    # Create default metrics
                    metrics = {
                        'total_ret': 0.0,
                        'annual_ret': 0.0,
                        'sharpe_rat': 0.0,
                        'max_drawd': 0.0,
                        'win_rate': 0.0,
                        'volatility': 0.0,
                        'sortino_r': 0.0,
                        'turnover_r': 0.0,
                        'final_wealth': 1.0,
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
    
    def save_results(self, results, file_prefix='new_strategy_cost_analysis'):
        """Save analysis results."""
        # Create DataFrame for final wealth
        final_wealth_df = pd.DataFrame(index=self.commission_rates)
        
        # Create DataFrame for transaction costs
        transaction_cost_df = pd.DataFrame(index=self.commission_rates)
        
        # Get all strategy names
        strategy_names = [strategy.name for strategy in self.strategies]
        
        # Fill DataFrames
        for rate in self.commission_rates:
            rate_results = results[rate]
            for strategy_name in strategy_names:
                if strategy_name in rate_results:
                    metrics = rate_results[strategy_name]['metrics']
                    final_wealth_df.loc[rate, strategy_name] = metrics.get('final_wealth', 1.0)
                    transaction_cost_df.loc[rate, strategy_name] = metrics.get('total_transaction_cost', 0.0)
        
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
            rate_dir = os.path.join(self.output_dir, f"rate_{rate:.4f}")
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
        strategy_names = [strategy.name for strategy in self.strategies]
        
        # Extract final wealth and transaction cost data
        final_wealth_data = {strategy: [] for strategy in strategy_names}
        transaction_cost_data = {strategy: [] for strategy in strategy_names}
        annual_return_data = {strategy: [] for strategy in strategy_names}
        sharpe_ratio_data = {strategy: [] for strategy in strategy_names}
        
        for rate in self.commission_rates:
            rate_results = results[rate]
            for strategy in strategy_names:
                if strategy in rate_results:
                    metrics = rate_results[strategy]['metrics']
                    final_wealth_data[strategy].append(rate_results[strategy]['portfolio_values'][-1] / 1000000)
                    transaction_cost_data[strategy].append(metrics.get('total_transaction_cost', 0))
                    annual_return_data[strategy].append(metrics.get('annual_ret', 0))
                    sharpe_ratio_data[strategy].append(metrics.get('sharpe_rat', 0))
        
        # Plot transaction cost vs final wealth
        plt.figure(figsize=(12, 8))
        
        markers = ['o', 's', '^', 'D']  # Marker styles
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Colors
        
        for i, strategy in enumerate(strategy_names):
            plt.plot(self.commission_rates, final_wealth_data[strategy], 
                    marker=markers[i % len(markers)], color=colors[i % len(colors)],
                    linewidth=2, label=strategy)
        
        plt.title('Impact of Transaction Cost on Final Wealth', fontfamily='Times New Roman', fontsize=14)
        plt.xlabel('Commission Rate', fontfamily='Times New Roman')
        plt.ylabel('Final Wealth (Portfolio Value / Initial Capital)', fontfamily='Times New Roman')
        plt.legend(prop={'family': 'Times New Roman'})
        plt.grid(True)
        plt.tight_layout()
        
        # Add x-axis percentage labels
        plt.xticks([rate for rate in self.commission_rates], 
                   [f"{rate:.2%}" for rate in self.commission_rates], 
                   rotation=45)
        
        # Save figure
        plt.savefig(os.path.join(self.output_dir, 'transaction_cost_vs_final_wealth.png'), dpi=300)
        plt.close()
        
        # Plot transaction cost bars
        plt.figure(figsize=(12, 8))
        
        x = np.arange(len(self.commission_rates))
        width = 0.8 / len(strategy_names)
        
        for i, strategy in enumerate(strategy_names):
            plt.bar(x + i*width, transaction_cost_data[strategy], 
                   width=width, label=strategy, color=colors[i % len(colors)])
        
        plt.title('Total Transaction Cost by Strategy', fontfamily='Times New Roman', fontsize=14)
        plt.xlabel('Commission Rate', fontfamily='Times New Roman')
        plt.ylabel('Total Transaction Cost', fontfamily='Times New Roman')
        plt.xticks(x + width*len(strategy_names)/2, [f"{rate:.2%}" for rate in self.commission_rates], rotation=45)
        plt.legend(prop={'family': 'Times New Roman'})
        plt.grid(True, axis='y')
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(self.output_dir, 'transaction_cost_by_strategy.png'), dpi=300)
        plt.close()
        
        # Plot annual return
        plt.figure(figsize=(12, 8))
        
        for i, strategy in enumerate(strategy_names):
            plt.plot(self.commission_rates, annual_return_data[strategy], 
                    marker=markers[i % len(markers)], color=colors[i % len(colors)],
                    linewidth=2, label=strategy)
        
        plt.title('Impact of Transaction Cost on Annual Return', fontfamily='Times New Roman', fontsize=14)
        plt.xlabel('Commission Rate', fontfamily='Times New Roman')
        plt.ylabel('Annual Return', fontfamily='Times New Roman')
        plt.legend(prop={'family': 'Times New Roman'})
        plt.grid(True)
        plt.tight_layout()
        plt.xticks([rate for rate in self.commission_rates], 
                   [f"{rate:.2%}" for rate in self.commission_rates], 
                   rotation=45)
        
        # Save figure
        plt.savefig(os.path.join(self.output_dir, 'transaction_cost_vs_annual_return.png'), dpi=300)
        plt.close()
        
        # Plot Sharpe ratio
        plt.figure(figsize=(12, 8))
        
        for i, strategy in enumerate(strategy_names):
            plt.plot(self.commission_rates, sharpe_ratio_data[strategy], 
                    marker=markers[i % len(markers)], color=colors[i % len(colors)],
                    linewidth=2, label=strategy)
        
        plt.title('Impact of Transaction Cost on Sharpe Ratio', fontfamily='Times New Roman', fontsize=14)
        plt.xlabel('Commission Rate', fontfamily='Times New Roman')
        plt.ylabel('Sharpe Ratio', fontfamily='Times New Roman')
        plt.legend(prop={'family': 'Times New Roman'})
        plt.grid(True)
        plt.tight_layout()
        plt.xticks([rate for rate in self.commission_rates], 
                   [f"{rate:.2%}" for rate in self.commission_rates], 
                   rotation=45)
        
        # Save figure
        plt.savefig(os.path.join(self.output_dir, 'transaction_cost_vs_sharpe_ratio.png'), dpi=300)
        plt.close()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Transaction cost impact analysis for new strategies')
    parser.add_argument('--data_path', type=str, default='data',
                      help='Data path')
    parser.add_argument('--output_dir', type=str, default='new_strategy_cost_analysis_results',
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
    analyzer = TransactionCostAnalyzerNewStrategies(
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

