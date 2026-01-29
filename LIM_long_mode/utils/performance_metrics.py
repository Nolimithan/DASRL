"""
Performance metrics module.

Provides portfolio performance evaluation metrics.
"""

import numpy as np
import pandas as pd
from scipy import stats

def calculate_performance_metrics(portfolio_values, trading_actions=None, risk_free_rate=0.04/252, benchmark_returns=None):
    """Compute portfolio performance metrics."""
    try:
        # Ensure input is numpy array
        portfolio_values = np.array(portfolio_values, dtype=np.float64)
        trading_days = len(portfolio_values)
        
        # 1. Total return
        total_return = ((portfolio_values[-1] / portfolio_values[0]) - 1) * 100
        
        # 2. Annualized return
        years = trading_days / 252  # Assume 252 trading days/year
        annual_return = ((1 + total_return/100) ** (1/years) - 1) * 100
        
        # 3. Daily returns
        daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        # 4. Annualized volatility
        volatility = np.std(daily_returns) * np.sqrt(252) * 100
        
        # 5. Sharpe ratio
        excess_returns = daily_returns - risk_free_rate
        sharpe_ratio = np.mean(excess_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0
        
        # 6. Max drawdown
        max_drawdown = calculate_max_drawdown(portfolio_values)
        
        # 7. Sortino ratio (downside risk only)
        downside_returns = daily_returns[daily_returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else np.std(daily_returns)
        sortino_ratio = np.mean(excess_returns) / downside_std * np.sqrt(252) if downside_std > 0 else 0
        
        # 8. Calmar ratio
        calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0
        
        # 9. Win rate (if trading actions provided)
        win_rate = 0
        if trading_actions is not None and len(trading_actions) > 0:
            trading_actions = np.array(trading_actions, dtype=np.int32)
            win_rate = calculate_win_rate(portfolio_values, trading_actions)
        
        # 10. Information ratio (if benchmark provided)
        information_ratio = 0
        if benchmark_returns is not None and len(benchmark_returns) >= len(daily_returns):
            # Align benchmark length
            benchmark_returns = benchmark_returns[:len(daily_returns)]
            # Compute excess returns
            excess_returns_vs_benchmark = daily_returns - benchmark_returns
            # Compute information ratio
            tracking_error = np.std(excess_returns_vs_benchmark)
            information_ratio = np.mean(excess_returns_vs_benchmark) / tracking_error * np.sqrt(252) if tracking_error > 0 else 0
        
        # 11. Skewness and kurtosis
        skewness = stats.skew(daily_returns) if len(daily_returns) > 2 else 0
        kurtosis = stats.kurtosis(daily_returns) if len(daily_returns) > 2 else 0
        
        # 12. Max consecutive up/down days
        pos_returns = daily_returns > 0
        neg_returns = daily_returns < 0
        max_consecutive_wins = max_consecutive(pos_returns)
        max_consecutive_losses = max_consecutive(neg_returns)
        
        # Aggregate metrics
        metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'information_ratio': information_ratio,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses,
            'daily_return_mean': np.mean(daily_returns) * 100,
            'daily_return_std': np.std(daily_returns) * 100
        }
        
        return metrics
        
    except Exception as e:
        print(f"Error computing performance metrics: {str(e)}")
        return {}

def calculate_max_drawdown(portfolio_values):
    """Compute maximum drawdown."""
    # Ensure correct data type
    portfolio_values = np.array(portfolio_values, dtype=np.float64)
    
    # Compute cumulative maximum
    peak_values = np.maximum.accumulate(portfolio_values)
    
    # Compute drawdown
    drawdowns = (peak_values - portfolio_values) / peak_values
    
    # Return max drawdown
    max_drawdown = np.max(drawdowns) * 100 if len(drawdowns) > 0 else 0
    
    return max_drawdown

def calculate_win_rate(portfolio_values, trading_actions):
    """Compute trade win rate."""
    if len(trading_actions) <= 1:
        return 0
    
    # Mark dates with trades
    trade_days = np.where(trading_actions != 0)[0]
    
    if len(trade_days) <= 1:
        return 0
    
    # Compute returns during trade periods
    profits = []
    
    for i in range(len(trade_days) - 1):
        start_idx = trade_days[i]
        end_idx = trade_days[i+1]
        
        if start_idx >= len(portfolio_values) or end_idx >= len(portfolio_values):
            continue
            
        profit = (portfolio_values[end_idx] / portfolio_values[start_idx]) - 1
        profits.append(profit)
    
    # Compute win rate
    if profits:
        wins = sum(1 for p in profits if p > 0)
        win_rate = (wins / len(profits)) * 100
    else:
        win_rate = 0
    
    return win_rate

def max_consecutive(bool_array):
    """Compute max consecutive True values."""
    if not any(bool_array):
        return 0
        
    # Convert bool array to 0/1
    int_array = np.array(bool_array, dtype=int)
    
    # Compute lengths of consecutive 1s
    consecutive_lengths = []
    current_length = 0
    
    for val in int_array:
        if val == 1:
            current_length += 1
        else:
            if current_length > 0:
                consecutive_lengths.append(current_length)
                current_length = 0
    
    # Handle trailing segment
    if current_length > 0:
        consecutive_lengths.append(current_length)
    
    # Return max length
    return max(consecutive_lengths) if consecutive_lengths else 0 

