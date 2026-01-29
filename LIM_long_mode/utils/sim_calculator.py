"""
SIM calculator module.

Compute SIM (Intraday Loss Accumulation) to measure short-term loss potential for short strategies.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta


class SIMCalculator:
    """SIM calculator for short-side stock selection."""
    
    def __init__(self, window_size=20, alpha=0.5, lookback_days=90):
        """Initialize the SIM calculator."""
        self.window_size = window_size
        self.alpha = alpha
        self.lookback_days = lookback_days
    
    def calculate_SIM_star(self, stock_data, current_date, lookback_days=90, commission_rate=0.005):
        """Compute SIM* for short strategies."""
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
            
            # Step 1 & 2: compute daily loss ratio, filter losing trades
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
            close_prices = np.where(close_prices <= 0, 1.0, close_prices)
            
            # Intraday loss ratio: Ot/(Ct(1-F)^2)
            # For shorts, higher open and lower close is better
            daily_returns = open_prices / (close_prices * (1 - commission_rate)**2)
            
            # Step 2: keep losing trades (ratio >= 1.0)
            loss_mask = daily_returns >= 1.0
            loss_returns = daily_returns[loss_mask]
            
            # Step 3 & 4: accumulate losses to get SIM
            if len(loss_returns) > 0:
                sim = np.prod(loss_returns)
            else:
                sim = 1.0  # If no losing trades, SIM is 1.0
            
            return sim
        
        except Exception as e:
            print(f"Error computing SIM: {e}")
            return 1.0
    
    def get_top_stocks(self, all_stock_data, benchmark_return, date, top_n=8, lim_calculator=None, risk_free_rate=0.0):
        """Get top stocks by SIM* (short potential)."""
        try:
            # Use lookback_days to get enough data
            lookback_days = self.lookback_days
            
            # Store SIM per stock
            stock_scores = {}
            
            # Compute SIM for each stock
            for symbol, data in all_stock_data.items():
                try:
                    # Ensure DataFrame
                    if not isinstance(data, pd.DataFrame):
                        continue
                    
                    # Compute SIM (short potential)
                    SIM_value = self.calculate_SIM_star(data, date, lookback_days)
                    
                    # Keep valid SIM values
                    if SIM_value > 0 and np.isfinite(SIM_value):
                        stock_scores[symbol] = SIM_value
                    
                except Exception as e:
                    print(f"Error processing stock {symbol}: {e}")
                    continue
            
            # Return empty list if no valid SIM
            if not stock_scores:
                print("No valid SIM values computed")
                return []
            
            # Sort and take top stocks
            sorted_stocks = sorted(stock_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Return top stocks
            result = [symbol for symbol, _ in sorted_stocks[:top_n]]
            print(f"SIM selected {len(result)} stocks for shorting")
            return result
        
        except Exception as e:
            print(f"Error getting top stocks: {e}")
            return []
    
    def get_SIM_percentile(self, all_stock_data, symbol, date, benchmark_return=None):
        """Get SIM* percentile for a stock on a date."""
        try:
            # Convert date to datetime
            if isinstance(date, str):
                target_date = pd.to_datetime(date)
            else:
                target_date = date
            
            # Use lookback_days to get enough data
            lookback_days = self.lookback_days
            
            # Compute SIM for all stocks
            all_SIM_values = {}
            for stock_symbol, data in all_stock_data.items():
                try:
                    # Ensure DataFrame
                    if not isinstance(data, pd.DataFrame):
                        continue
                    
                    # Compute SIM
                    SIM_value = self.calculate_SIM_star(data, target_date, lookback_days)
                    
                    # Validate SIM value
                    if SIM_value > 0 and np.isfinite(SIM_value):
                        all_SIM_values[stock_symbol] = SIM_value
                except Exception as e:
                    print(f"Error computing SIM for {stock_symbol}: {e}")
                    continue
            
            # Return 0 if no valid SIM values
            if not all_SIM_values:
                return 0.0
            
            # Return 0 if target stock missing
            if symbol not in all_SIM_values:
                return 0.0
            
            # Compute percentile
            SIM_values_list = list(all_SIM_values.values())
            SIM_value = all_SIM_values[symbol]
            percentile = sum(v <= SIM_value for v in SIM_values_list) / len(SIM_values_list)
            
            return percentile
        
        except Exception as e:
            print(f"Error computing SIM percentile: {e}")
            return 0.0
    
    def calculate_SIM_series(self, data, benchmark_returns=None, start_date=None, end_date=None):
        """Compute a SIM series over a date range."""
        try:
            if start_date is None and end_date is None:
                # Use full dataset range
                if 'date' in data.columns:
                    dates = pd.to_datetime(data['date'])
                else:
                    dates = data.index
            else:
                # Use specified date range
                if 'date' in data.columns:
                    date_col = pd.to_datetime(data['date'])
                    date_mask = np.ones(len(date_col), dtype=bool)
                    
                    if start_date is not None:
                        start_date = pd.to_datetime(start_date)
                        date_mask &= (date_col >= start_date)
                        
                    if end_date is not None:
                        end_date = pd.to_datetime(end_date)
                        date_mask &= (date_col <= end_date)
                        
                    dates = date_col[date_mask]
                    data = data[date_mask].copy()
                else:
                    # Filter by index dates
                    if start_date is not None:
                        start_date = pd.to_datetime(start_date)
                    if end_date is not None:
                        end_date = pd.to_datetime(end_date)
                    
                    data = data.loc[start_date:end_date].copy() if start_date or end_date else data
                    dates = data.index
            
            # Initialize SIM series
            SIM_values = []
            dates_list = []
            
            # Compute SIM for each date
            for date in dates:
                try:
                    SIM_value = self.calculate_SIM_star(data, date, self.lookback_days)
                    SIM_values.append(SIM_value)
                    dates_list.append(date)
                except Exception as e:
                    print(f"Error computing SIM for date {date}: {e}")
            
            # Build series with date index
            if SIM_values:
                return pd.Series(SIM_values, index=dates_list)
            else:
                return pd.Series()
        
        except Exception as e:
            print(f"Error computing SIM series: {e}")
            return pd.Series()
    
    def get_sorted_stocks(self, all_stock_data, benchmark_return, date, lookback_days=None, lim_calculator=None, risk_free_rate=0.0):
        """Return stocks sorted by SIM (full list)."""
        try:
            # Use provided lookback_days or default
            actual_lookback_days = lookback_days if lookback_days is not None else self.lookback_days
            
            # Store SIM per stock
            stock_scores = {}
            
            # Compute SIM for each stock
            for symbol, data in all_stock_data.items():
                try:
                    # Ensure DataFrame
                    if not isinstance(data, pd.DataFrame):
                        continue
                    
                    # Compute SIM (short potential)
                    SIM_value = self.calculate_SIM_star(data, date, actual_lookback_days)
                    
                    # Keep valid SIM values
                    if SIM_value > 0 and np.isfinite(SIM_value):
                        stock_scores[symbol] = SIM_value
                    
                except Exception as e:
                    print(f"Error processing stock {symbol}: {e}")
                    continue
            
            # Return empty list if no valid SIM
            if not stock_scores:
                print("No valid SIM values computed")
                return []
            
            # Sort all stocks
            sorted_stocks = sorted(stock_scores.items(), key=lambda x: x[1], reverse=True)
            
            return sorted_stocks
        
        except Exception as e:
            print(f"Error getting sorted stocks: {e}")
            return []
    
    def calculate_LIM_star(self, stock_data, current_date, lookback_days=90, commission_rate=0.005):
        """[Deprecated] LIM* calculation (unused)."""
        print("Warning: calculate_LIM_star is deprecated; LIM not used")
        return 0.0  # Return 0 to avoid use in adjustments
    
    def calculate_risk_adjusted_score(self, stock_data, current_date, LIM_value=None, lim_calculator=None, lookback_days=90, commission_rate=0.005, risk_free_rate=0.0):
        """[Deprecated] Risk-adjusted short score (unused)."""
        print("Warning: calculate_risk_adjusted_score is deprecated; SIM only")
        try:
            # Compute SIM (short potential)
            SIM_value = self.calculate_SIM_star(stock_data, current_date, lookback_days, commission_rate)
            
            # Return raw SIM value
            return SIM_value
        
        except Exception as e:
            print(f"Error computing risk-adjusted score: {e}")
            return 0.0 

