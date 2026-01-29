"""
LIM calculator module.

Compute LIM (Intraday Return Accumulation) to measure short-term return potential.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta


class LIMCalculator:
    """LIM calculator for ranking stocks by LIM."""
    
    def __init__(self, window_size=20, alpha=0.5, lookback_days=90):
        """Initialize the LIM calculator.

        Args:
            window_size: Rolling window size for LIM.
            alpha: Risk aversion coefficient for weighting.
            lookback_days: Lookback days to ensure enough history.
        """
        self.window_size = window_size
        self.alpha = alpha
        self.lookback_days = lookback_days
    
    def calculate_LIM_star(self, stock_data, current_date, lookback_days=90, commission_rate=0.005):
        """Compute LIM* for a stock using the paper's steps.

        Args:
            stock_data: OHLC DataFrame.
            current_date: Current date.
            lookback_days: Lookback window (default 90).
            commission_rate: Commission rate (default 0.005).

        Returns:
            float: LIM* value.
        """
        try:
            # Convert date to datetime
            if isinstance(current_date, str):
                current_date = pd.to_datetime(current_date)
            
            # Ensure data has date index
            if 'date' in stock_data.columns:
                temp_data = stock_data.copy()
                temp_data['date'] = pd.to_datetime(temp_data['date'])
                # Get window data
                window_start_date = current_date - pd.Timedelta(days=lookback_days)
                window_data = temp_data[(temp_data['date'] >= window_start_date) & (temp_data['date'] <= current_date)]
            else:
                # Assume index is already dates
                window_start_date = current_date - pd.Timedelta(days=lookback_days)
                window_data = stock_data[(stock_data.index >= window_start_date) & (stock_data.index <= current_date)]
            
            # Ensure enough data points
            if len(window_data) < 2:
                return 0.0
            
            # Step 1 & 2: compute daily returns and filter profitable trades
            # Ensure open/close columns exist
            if ('open' in window_data.columns and 'close' in window_data.columns):
                open_prices = window_data['open'].values
                close_prices = window_data['close'].values
            elif ('Open' in window_data.columns and 'Close' in window_data.columns):
                open_prices = window_data['Open'].values
                close_prices = window_data['Close'].values
            else:
                # Try raw price columns
                if ('raw_Open' in window_data.columns and 'raw_Close' in window_data.columns):
                    open_prices = window_data['raw_Open'].values
                    close_prices = window_data['raw_Close'].values
                else:
                    return 0.0
            
            # Handle invalid values
            open_prices = np.nan_to_num(open_prices, nan=1.0, posinf=1.0, neginf=1.0)
            close_prices = np.nan_to_num(close_prices, nan=1.0, posinf=1.0, neginf=1.0)
            
            # Avoid zero prices
            open_prices = np.where(open_prices <= 0, 1.0, open_prices)
            
            # Intraday return: Ct(1-F)^2/Ot
            daily_returns = (close_prices * (1 - commission_rate)**2) / open_prices
            
            # Step 2: keep profitable trades (>= 1.0)
            profitable_mask = daily_returns >= 1.0
            profitable_returns = daily_returns[profitable_mask]
            
            # Step 3 & 4: accumulate profitable returns to get LIM
            if len(profitable_returns) > 0:
                lim = np.prod(profitable_returns)
            else:
                lim = 1.0  # If no profitable trades, LIM is 1.0
            
            return lim
        
        except Exception as e:
            print(f"Error computing LIM: {e}")
            return 1.0
    
    def get_top_stocks(self, all_stock_data, benchmark_return, date, top_n=8):
        """Select top stocks by LIM* for a given date."""
        try:
            # Determine window start date with lookback
            lookback_days = self.lookback_days
            
            # Store LIM values per stock
            stock_LIM_values = {}
            
            # Compute LIM for each stock
            for symbol, data in all_stock_data.items():
                try:
                    # Ensure DataFrame
                    if not isinstance(data, pd.DataFrame):
                        continue
                    
                    # Compute LIM
                    LIM_value = self.calculate_LIM_star(data, date, lookback_days)
                    
                    # Keep valid LIM values
                    if LIM_value > 0 and np.isfinite(LIM_value):
                        stock_LIM_values[symbol] = LIM_value
                    
                except Exception as e:
                    print(f"Error processing stock {symbol}: {e}")
                    continue
            
            # Return empty if no valid LIM values
            if not stock_LIM_values:
                print("No valid LIM values computed")
                return []
            
            # Step 5: normalize to LIM*
            # Find max LIM
            LIM_max = max(stock_LIM_values.values())
            
            # Normalize LIM values to LIM*
            stock_LIM_star = {symbol: (lim/LIM_max * 100) for symbol, lim in stock_LIM_values.items()}
            
            # Sort and take top stocks
            sorted_stocks = sorted(stock_LIM_star.items(), key=lambda x: x[1], reverse=True)
            
            # Return top stocks
            result = [symbol for symbol, _ in sorted_stocks[:top_n]]
            print(f"lim* selected {len(result)} stocks")
            return result
        
        except Exception as e:
            print(f"Error getting top stocks: {e}")
            return []
    
    def get_LIM_percentile(self, all_stock_data, symbol, date, benchmark_return=None):
        """Get LIM* percentile for a stock on a date."""
        try:
            # Convert date to datetime
            if isinstance(date, str):
                target_date = pd.to_datetime(date)
            else:
                target_date = date
            
            # Use lookback_days to get enough data
            lookback_days = self.lookback_days
            
            # Compute LIM for all stocks
            all_LIM_values = {}
            for stock_symbol, data in all_stock_data.items():
                try:
                    # Ensure DataFrame
                    if not isinstance(data, pd.DataFrame):
                        continue
                    
                    # Compute LIM
                    LIM_value = self.calculate_LIM_star(data, target_date, lookback_days)
                    
                    # Validate LIM value
                    if LIM_value > 0 and np.isfinite(LIM_value):
                        all_LIM_values[stock_symbol] = LIM_value
                except Exception as e:
                    print(f"Error computing LIM for {stock_symbol}: {e}")
                    continue
            
            # Return 0 if no valid LIM values
            if not all_LIM_values:
                return 0.0
            
            # Return 0 if target stock missing
            if symbol not in all_LIM_values:
                return 0.0
            
            # Compute percentile
            LIM_values_list = list(all_LIM_values.values())
            LIM_value = all_LIM_values[symbol]
            percentile = sum(v <= LIM_value for v in LIM_values_list) / len(LIM_values_list)
            
            return percentile
        
        except Exception as e:
            print(f"Error computing LIM percentile: {e}")
            return 0.0

    # Keep legacy calculate_LIM for backward compatibility
    def calculate_LIM(self, stock_return, benchmark_return):
        """Legacy LIM method kept for backward compatibility."""
        try:
            # Ensure aligned lengths
            min_length = min(len(stock_return), len(benchmark_return))
            if min_length < 2:
                return 0
            
            # Use most recent min_length points
            stock_return = stock_return[-min_length:]
            benchmark_return = benchmark_return[-min_length:]
            
            # Handle NaN and inf
            stock_return = np.nan_to_num(stock_return, nan=0.0, posinf=0.0, neginf=0.0)
            benchmark_return = np.nan_to_num(benchmark_return, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Compute excess returns
            excess_return = stock_return - benchmark_return
            
            # Compute mean excess return
            mean_excess_return = np.mean(excess_return)
            
            # Compute excess return volatility
            std_excess_return = np.std(excess_return)
            
            # Avoid division by zero
            if std_excess_return < 1e-6:
                return 0
            
            # Compute LIM
            lim = mean_excess_return / (std_excess_return ** self.alpha)
            
            # Ensure finite result
            if not np.isfinite(lim):
                return 0
            
            return lim
        
        except Exception as e:
            print(f"Error computing LIM: {e}")
            return 0
    
    def calculate_LIM_series(self, data, benchmark_returns=None, start_date=None, end_date=None):
        """Compute LIM* series over a date range."""
        try:
            # Ensure required columns exist
            required_price_pairs = [
                ('open', 'close'),
                ('Open', 'Close'),
                ('raw_Open', 'raw_Close')
            ]
            
            has_required_columns = False
            for open_col, close_col in required_price_pairs:
                if open_col in data.columns and close_col in data.columns:
                    has_required_columns = True
                    break
                
            if not has_required_columns:
                print("Data missing required open/close columns")
                return pd.Series(dtype=float)
            
            # Ensure date column or date index exists
            if 'date' in data.columns:
                # Ensure date column is datetime
                date_col = pd.to_datetime(data['date'])
                
                # Copy to avoid mutating original data
                temp_data = data.copy()
                temp_data['date'] = date_col
                
                # Filter by date range if provided
                if start_date is not None:
                    start_date = pd.to_datetime(start_date)
                    temp_data = temp_data[temp_data['date'] >= start_date]
                if end_date is not None:
                    end_date = pd.to_datetime(end_date)
                    temp_data = temp_data[temp_data['date'] <= end_date]
                    
                # Get all dates
                dates = sorted(temp_data['date'].unique())
            elif isinstance(data.index, pd.DatetimeIndex):
                # If index is already date-like
                temp_data = data.copy()
                
                # Filter by date range if provided
                if start_date is not None:
                    start_date = pd.to_datetime(start_date)
                    temp_data = temp_data[temp_data.index >= start_date]
                if end_date is not None:
                    end_date = pd.to_datetime(end_date)
                    temp_data = temp_data[temp_data.index <= end_date]
                    
                # Get all dates
                dates = sorted(temp_data.index)
            else:
                print("Data has no valid date column or index")
                return pd.Series(dtype=float)
            
            # Initialize result series
            LIM_values = pd.Series(index=dates, dtype=float)
            lookback_days = self.lookback_days
            
            # Compute LIM* for each date
            all_LIM_values = []
            for day in dates:
                # Compute LIM
                LIM_value = self.calculate_LIM_star(temp_data, day, lookback_days)
                # Store result
                all_LIM_values.append(LIM_value)
            
            # Build result series
            LIM_values = pd.Series(all_LIM_values, index=dates)
            
            # Normalize LIM to LIM*
            # Find max LIM value
            LIM_max = LIM_values.max()
            if LIM_max > 0:
                # Normalize to LIM* (0-100%)
                LIM_star = LIM_values / LIM_max * 100
            else:
                LIM_star = LIM_values * 0
            
            return LIM_star
            
        except Exception as e:
            print(f"Error computing LIM* series: {e}")
            return pd.Series(dtype=float)

    def get_sorted_stocks(self, all_stock_data, benchmark_return, date, lookback_days=None):
        """Get stocks sorted by LIM*."""
        try:
            # Use instance lookback_days or provided value
            if lookback_days is None:
                lookback_days = self.lookback_days
            
            # Store LIM per stock
            stock_LIM_values = {}
            
            # Compute LIM for each stock
            for symbol, data in all_stock_data.items():
                try:
                    # Ensure DataFrame
                    if not isinstance(data, pd.DataFrame):
                        continue
                    
                    # Compute LIM
                    LIM_value = self.calculate_LIM_star(data, date, lookback_days)
                    
                    # Keep valid LIM values
                    if LIM_value > 0 and np.isfinite(LIM_value):
                        stock_LIM_values[symbol] = LIM_value
                    
                except Exception as e:
                    print(f"Error processing stock {symbol}: {e}")
                    continue
            
            # Return empty list if no valid LIM
            if not stock_LIM_values:
                print("No valid LIM values computed")
                return []
            
            # Step 5: normalize LIM to LIM*
            # Find max LIM value
            LIM_max = max(stock_LIM_values.values())
            
            # Normalize LIM to LIM*
            stock_LIM_star = {symbol: (lim/LIM_max * 100) for symbol, lim in stock_LIM_values.items()}
            
            # Sort and collect all stocks
            sorted_stocks = sorted(stock_LIM_star.items(), key=lambda x: x[1], reverse=True)
            
            # Return tickers sorted by LIM*
            return [symbol for symbol, _ in sorted_stocks]
            
        except Exception as e:
            print(f"Error getting sorted stocks: {e}")
            return [] 

