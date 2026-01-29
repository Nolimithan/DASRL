"""
Portfolio strategy comparison module.

Compares multiple OLPS (Online Portfolio Selection) strategies on A-share
semiconductor stocks.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
from tqdm import tqdm
import yaml
from utils.data_loader_factory import create_data_loader
from config.config import get_config
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.optim import Adam

# Set global font to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign rendering

# Strategy evaluation metrics
def calculate_metrics(portfolio_values, daily_weights_changes=None):
    """Compute portfolio performance metrics."""
    # Validate input data
    if np.any(np.isnan(portfolio_values)) or np.any(np.isinf(portfolio_values)):
        print("Warning: portfolio values contain NaN/inf; attempting fix")
        # Attempt to fix invalid values
        portfolio_values = np.array(portfolio_values)
        mask = ~(np.isnan(portfolio_values) | np.isinf(portfolio_values))
        if not np.any(mask):
            print("Error: cannot fix portfolio values; all values invalid")
            return {
                'total_ret': 0.0,
                'annual_ret': 0.0,
                'sharpe_rat': 0.0,
                'max_drawd': 0.0,
                'win_rate': 0.0,
                'volatility': 0.0,
                'sortino_r': 0.0,
                'turnover_r': 0.0
            }
        
        # Replace invalid values with valid ones
        valid_values = portfolio_values[mask]
        if len(valid_values) == 0:
            return {
                'total_ret': 0.0,
                'annual_ret': 0.0,
                'sharpe_rat': 0.0,
                'max_drawd': 0.0,
                'win_rate': 0.0,
                'volatility': 0.0,
                'sortino_r': 0.0,
                'turnover_r': 0.0
            }
        
        first_valid = valid_values[0]
        for i in range(len(portfolio_values)):
            if np.isnan(portfolio_values[i]) or np.isinf(portfolio_values[i]):
                portfolio_values[i] = first_valid
            else:
                first_valid = portfolio_values[i]
    
    try:
        # Daily returns
        daily_returns = portfolio_values[1:] / portfolio_values[:-1] - 1
        
        # Handle invalid daily returns
        daily_returns = np.nan_to_num(daily_returns, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 1. Total return
        total_ret = portfolio_values[-1] / portfolio_values[0] - 1
        
        # 2. Annualized return
        days = len(portfolio_values)
        annual_ret = (1 + total_ret) ** (252 / days) - 1
        
        # 3. Volatility
        volatility = np.std(daily_returns) * np.sqrt(252)
        
        # 4. Sharpe ratio (assume 4% risk-free rate)
        rf = 0.04
        daily_rf = (1 + rf) ** (1/252) - 1
        if volatility == 0:
            print("Warning: portfolio volatility is 0; cannot compute Sharpe ratio")
            sharpe_rat = 0
        else:
            sharpe_rat = (np.mean(daily_returns) - daily_rf) / np.std(daily_returns) * np.sqrt(252)
        
        # 5. Maximum drawdown
        hwm = np.zeros_like(portfolio_values)
        drawdowns = np.zeros_like(portfolio_values)
        for t in range(1, len(portfolio_values)):
            hwm[t] = max(hwm[t-1], portfolio_values[t])
            if hwm[t] == 0:
                drawdowns[t] = 0
            else:
                drawdowns[t] = (hwm[t] - portfolio_values[t]) / hwm[t]
        max_drawd = max(drawdowns)
        
        # 6. Win rate (positive-return days / total days)
        win_rate = np.sum(daily_returns > 0) / len(daily_returns)
        
        # 7. Sortino ratio (downside risk)
        # Use std of negative returns as downside risk
        downside_returns = daily_returns[daily_returns < 0]
        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            sortino_r = 0 if np.mean(daily_returns) <= daily_rf else float('inf')
        else:
            sortino_r = (np.mean(daily_returns) - daily_rf) / np.std(downside_returns) * np.sqrt(252)
        
        # 8. Turnover rate - set to 0 per requirement
        turnover_r = 0.0
        
        # Validate results
        metrics = {
            'total_ret': total_ret,
            'annual_ret': annual_ret,
            'sharpe_rat': sharpe_rat,
            'max_drawd': max_drawd,
            'win_rate': win_rate,
            'volatility': volatility,
            'sortino_r': sortino_r,
            'turnover_r': turnover_r
        }
        
        # Check and fix invalid values
        for key, value in metrics.items():
            if np.isnan(value) or np.isinf(value):
                print(f"Warning: invalid {key}, using default 0.0")
                metrics[key] = 0.0
        
        return metrics
    except Exception as e:
        print(f"Error computing portfolio metrics: {str(e)}")
        return {
            'total_ret': 0.0,
            'annual_ret': 0.0,
            'sharpe_rat': 0.0,
            'max_drawd': 0.0,
            'win_rate': 0.0,
            'volatility': 0.0,
            'sortino_r': 0.0,
            'turnover_r': 0.0
        }

# Base strategy class
class BaseStrategy:
    """Base portfolio strategy."""
    
    def __init__(self, name):
        self.name = name
    
    def allocate(self, data, current_weights=None):
        """Compute portfolio weights."""
        raise NotImplementedError("Subclasses must implement this method")

# Buy and Hold strategy
class UBAH(BaseStrategy):
    """Uniform buy-and-hold strategy."""
    
    def __init__(self):
        super().__init__("UBAH")
    
    def allocate(self, data, current_weights=None):
        """Allocate equal weights without rebalancing."""
        n_assets = data.shape[1]
        return np.ones(n_assets) / n_assets

# Uniform constant rebalanced strategy
class UCRP(BaseStrategy):
    """Uniform constant rebalanced portfolio."""
    
    def __init__(self):
        super().__init__("UCRP")
    
    def allocate(self, data, current_weights=None):
        """Allocate equal weights."""
        n_assets = data.shape[1]
        return np.ones(n_assets) / n_assets

# Exponentiated Gradient strategy
class EG(BaseStrategy):
    """Exponentiated gradient strategy."""
    
    def __init__(self, eta=0.05):
        super().__init__("EG")
        self.eta = eta
    
    def allocate(self, data, current_weights=None):
        """Allocate weights using exponentiated gradient."""
        n_assets = data.shape[1]
        
        if current_weights is None:
            return np.ones(n_assets) / n_assets
            
        # Get recent price relatives
        if len(data) < 2:
            return current_weights
            
        price_relatives = data.iloc[-1] / data.iloc[-2]
        
        # Update weights
        numerator = current_weights * np.exp(self.eta * price_relatives)
        return numerator / np.sum(numerator)

# Anticor strategy
class Anticor(BaseStrategy):
    """Anti-correlation strategy."""
    
    def __init__(self, window=30):
        super().__init__("Anticor")
        self.window = window
    
    def allocate(self, data, current_weights=None):
        """Allocate weights using the Anticor algorithm."""
        n_assets = data.shape[1]
        
        if len(data) < self.window * 2:
            return np.ones(n_assets) / n_assets
            
        # Compute log returns for two windows
        try:
            # Ensure same window size and aligned indices
            window1_end = len(data) - self.window
            window1_start = window1_end - self.window
            
            log_returns1 = np.log(data.iloc[window1_start+1:window1_end+1].values / 
                                data.iloc[window1_start:window1_end].values)
            
            window2_end = len(data)
            window2_start = window2_end - self.window
            
            log_returns2 = np.log(data.iloc[window2_start+1:window2_end].values / 
                                data.iloc[window2_start:window2_end-1].values)
            
            # Ensure window sizes match
            if log_returns1.shape[0] != log_returns2.shape[0]:
                print(f"Warning: window size mismatch ({log_returns1.shape[0]} vs {log_returns2.shape[0]})")
                # If mismatch, return uniform allocation
                return np.ones(n_assets) / n_assets
        
        except Exception as e:
            print(f"Anticor computation error: {str(e)}")
            return np.ones(n_assets) / n_assets
        
        # Compute mean and covariance
        mu1 = np.mean(log_returns1, axis=0)
        mu2 = np.mean(log_returns2, axis=0)
        
        # Simplified Anticor implementation
        weights = np.ones(n_assets) / n_assets
        return weights

# PAMR strategy
class PAMR(BaseStrategy):
    """Passive aggressive mean reversion strategy."""
    
    def __init__(self, epsilon=0.5, C=500):
        super().__init__("PAMR")
        self.epsilon = epsilon
        self.C = C
    
    def allocate(self, data, current_weights=None):
        """Allocate weights using PAMR."""
        n_assets = data.shape[1]
        
        if current_weights is None or len(data) < 2:
            return np.ones(n_assets) / n_assets
            
        # Get recent price relatives
        price_relatives = data.iloc[-1] / data.iloc[-2]
        
        # Compute portfolio return
        portfolio_return = np.sum(current_weights * price_relatives)
        
        # Compute loss
        loss = max(0, portfolio_return - self.epsilon)
        
        # If no loss, keep current weights
        if loss == 0:
            return current_weights
            
        # Update weights
        tau = min(self.C, loss / np.sum((price_relatives - np.mean(price_relatives))**2))
        new_weights = current_weights - tau * (price_relatives - np.mean(price_relatives))
        
        # Project onto simplex
        new_weights = self._simplex_projection(new_weights)
        return new_weights
    
    def _simplex_projection(self, v):
        """Project vector onto simplex."""
        n = len(v)
        if np.sum(v) == 1 and np.all(v >= 0):
            return v
            
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        
        rho = 0
        for i in range(n):
            if u[i] + (1 - cssv[i]) / (i + 1) > 0:
                rho = i
                
        theta = (1 - cssv[rho]) / (rho + 1)
        return np.maximum(v + theta, 0)

# OLMAR strategy
class OLMAR(BaseStrategy):
    """Online moving average reversion strategy."""
    
    def __init__(self, epsilon=10, window=5):
        super().__init__("OLMAR")
        self.epsilon = epsilon
        self.window = window
    
    def allocate(self, data, current_weights=None):
        """Allocate weights using OLMAR."""
        n_assets = data.shape[1]
        
        if current_weights is None or len(data) < self.window:
            return np.ones(n_assets) / n_assets
            
        # Predict next-period prices
        last_prices = data.iloc[-self.window:].values
        predicted_prices = np.mean(last_prices, axis=0)
        
        # Compute predicted price relatives
        price_relatives = predicted_prices / data.iloc[-1].values
        
        # If all predictions equal, return uniform allocation
        if np.all(price_relatives == price_relatives[0]):
            return np.ones(n_assets) / n_assets
            
        # Apply OLMAR algorithm
        lam = max(0, (self.epsilon - np.dot(current_weights, price_relatives)) / 
                 np.sum((price_relatives - np.mean(price_relatives))**2))
        
        new_weights = current_weights + lam * (price_relatives - np.mean(price_relatives))
        
        # Project onto simplex
        new_weights = self._simplex_projection(new_weights)
        return new_weights
    
    def _simplex_projection(self, v):
        """Project vector onto simplex."""
        n = len(v)
        if np.sum(v) == 1 and np.all(v >= 0):
            return v
            
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        
        rho = 0
        for i in range(n):
            if u[i] + (1 - cssv[i]) / (i + 1) > 0:
                rho = i
                
        theta = (1 - cssv[rho]) / (rho + 1)
        return np.maximum(v + theta, 0)

# CORN-K strategy
class CORN_K(BaseStrategy):
    """Correlation-driven nonparametric learning (KNN) strategy."""
    
    def __init__(self, window=5, K=3, rho=10):
        super().__init__("CORN-K")
        self.window = window
        self.K = K
        self.rho = rho
    
    def allocate(self, data, current_weights=None):
        """Allocate weights using CORN-K."""
        n_assets = data.shape[1]
        
        if len(data) < self.window + 1:
            return np.ones(n_assets) / n_assets
            
        # Compute price relatives
        price_relatives = np.array([data.iloc[i+1] / data.iloc[i] for i in range(len(data)-1)])
        
        # Get latest price-relative window
        current_window = price_relatives[-self.window:]
        
        # Find similar historical windows
        simSIMr_indices = []
        for i in range(len(price_relatives) - self.window):
            historical_window = price_relatives[i:i+self.window]
            
            # Compute correlation
            correlation = np.corrcoef(current_window.T.flatten(), historical_window.T.flatten())[0, 1]
            
            if correlation > self.rho:
                simSIMr_indices.append(i + self.window)
        
        # If no similar windows, return uniform allocation
        if len(simSIMr_indices) < self.K:
            return np.ones(n_assets) / n_assets
        
        # Select K most similar windows
        if len(simSIMr_indices) > self.K:
            simSIMr_indices = simSIMr_indices[-self.K:]
        
        # Compute optimal weights
        A = np.zeros((n_assets, n_assets))
        b = np.zeros(n_assets)
        
        for idx in simSIMr_indices:
            x_t = price_relatives[idx]
            A += np.outer(x_t, x_t)
            b += x_t
        
        # Solve quadratic program
        try:
            weights = np.linalg.solve(A, b)
            weights = self._simplex_projection(weights)
        except:
            weights = np.ones(n_assets) / n_assets
        
        return weights
    
    def _simplex_projection(self, v):
        """Project vector onto simplex."""
        n = len(v)
        if np.sum(v) == 1 and np.all(v >= 0):
            return v
            
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        
        rho = 0
        for i in range(n):
            if u[i] + (1 - cssv[i]) / (i + 1) > 0:
                rho = i
                
        theta = (1 - cssv[rho]) / (rho + 1)
        return np.maximum(v + theta, 0)

# CWMR-Var strategy
class CWMR_Var(BaseStrategy):
    """Confidence-weighted mean reversion (variance version)."""
    
    def __init__(self, epsilon=0.5, phi=2.0):
        super().__init__("CWMR-Var")
        self.epsilon = epsilon
        self.phi = phi
    
    def allocate(self, data, current_weights=None):
        """Allocate weights using CWMR-Var."""
        n_assets = data.shape[1]
        
        if current_weights is None or len(data) < 2:
            return np.ones(n_assets) / n_assets
            
        # Initialize mean and covariance
        if not hasattr(self, 'mu') or not hasattr(self, 'sigma'):
            self.mu = current_weights
            self.sigma = np.eye(n_assets) / n_assets
        
        # Get recent price relatives
        price_relatives = data.iloc[-1] / data.iloc[-2]
        
        # Compute portfolio return
        portfolio_return = np.sum(self.mu * price_relatives)
        
        # Compute intermediate variables
        M = price_relatives
        V = np.dot(np.dot(M, self.sigma), M)
        x = max(0, 1 - self.epsilon - np.log(portfolio_return)) / V
        
        # Update weights and covariance
        self.sigma_inv = np.linalg.inv(self.sigma)
        
        # Compute new weights
        tmp1 = self.sigma_inv + 2 * x * np.outer(price_relatives, price_relatives)
        tmp2 = self.sigma_inv.dot(self.mu) - x * (price_relatives - portfolio_return * price_relatives)
        
        # Update covariance matrix
        self.sigma = np.linalg.inv(tmp1)
        
        # Update mean
        self.mu = self.sigma.dot(tmp2)
        
        # Project onto simplex
        self.mu = self._simplex_projection(self.mu)
        
        return self.mu
    
    def _simplex_projection(self, v):
        """Project vector onto simplex."""
        n = len(v)
        if np.sum(v) == 1 and np.all(v >= 0):
            return v
            
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        
        rho = 0
        for i in range(n):
            if u[i] + (1 - cssv[i]) / (i + 1) > 0:
                rho = i
                
        theta = (1 - cssv[rho]) / (rho + 1)
        return np.maximum(v + theta, 0)

# RMR strategy
class RMR(BaseStrategy):
    """Robust median reversion strategy."""
    
    def __init__(self, window=5, epsilon=10, beta=0.5):
        super().__init__("RMR")
        self.window = window
        self.epsilon = epsilon
        self.beta = beta
    
    def allocate(self, data, current_weights=None):
        """Allocate weights using RMR."""
        n_assets = data.shape[1]
        
        if current_weights is None or len(data) < self.window:
            return np.ones(n_assets) / n_assets
            
        # Compute price relatives
        price_relatives = np.array([data.iloc[i+1] / data.iloc[i] for i in range(len(data)-1)])
        
        # Get latest price-relative window
        window_price_relatives = price_relatives[-self.window:]
        
        # Compute median of price relatives
        median_price_relatives = np.median(window_price_relatives, axis=0)
        
        # Compute expected price relatives
        expected_relatives = self.beta * median_price_relatives + (1 - self.beta) * np.ones(n_assets)
        
        # If all predictions equal, return uniform allocation
        if np.all(expected_relatives == expected_relatives[0]):
            return np.ones(n_assets) / n_assets
            
        # Apply RMR algorithm
        lam = max(0, (self.epsilon - np.dot(current_weights, expected_relatives)) / 
                 np.sum((expected_relatives - np.mean(expected_relatives))**2))
        
        new_weights = current_weights + lam * (expected_relatives - np.mean(expected_relatives))
        
        # Project onto simplex
        new_weights = self._simplex_projection(new_weights)
        return new_weights
    
    def _simplex_projection(self, v):
        """Project vector onto simplex."""
        n = len(v)
        if np.sum(v) == 1 and np.all(v >= 0):
            return v
            
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        
        rho = 0
        for i in range(n):
            if u[i] + (1 - cssv[i]) / (i + 1) > 0:
                rho = i
                
        theta = (1 - cssv[rho]) / (rho + 1)
        return np.maximum(v + theta, 0)

# TCO1 strategy
class TCO1(BaseStrategy):
    """Transaction cost optimization strategy 1."""
    
    def __init__(self, window=5, epsilon=10, gamma=0.01):
        super().__init__("TCO1")
        self.window = window
        self.epsilon = epsilon
        self.gamma = gamma  # Transaction cost coefficient
    
    def allocate(self, data, current_weights=None):
        """Allocate weights using TCO1."""
        n_assets = data.shape[1]
        
        if current_weights is None or len(data) < self.window:
            return np.ones(n_assets) / n_assets
            
        # Compute moving average over recent window
        last_prices = data.iloc[-self.window:].values
        predicted_prices = np.mean(last_prices, axis=0)
        
        # Compute expected price relatives
        price_relatives = predicted_prices / data.iloc[-1].values
        
        # If all predictions equal, return uniform allocation
        if np.all(price_relatives == price_relatives[0]):
            return np.ones(n_assets) / n_assets
            
        # Compute OLMAR base weights
        lam = max(0, (self.epsilon - np.dot(current_weights, price_relatives)) / 
                 np.sum((price_relatives - np.mean(price_relatives))**2))
        
        olmar_weights = current_weights + lam * (price_relatives - np.mean(price_relatives))
        olmar_weights = self._simplex_projection(olmar_weights)
        
        # Adjust for transaction costs
        # TCO1 penalizes deviation from current weights to reduce turnover
        delta = olmar_weights - current_weights
        
        # L1 norm penalty
        l1_penalty = np.sum(np.abs(delta)) * self.gamma
        
        # Adjusted new weights
        new_weights = current_weights + delta * (1 - l1_penalty)
        if np.sum(new_weights) != 1 or np.any(new_weights < 0):
            new_weights = self._simplex_projection(new_weights)
        
        return new_weights
    
    def _simplex_projection(self, v):
        """Project vector onto simplex."""
        n = len(v)
        if np.sum(v) == 1 and np.all(v >= 0):
            return v
            
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        
        rho = 0
        for i in range(n):
            if u[i] + (1 - cssv[i]) / (i + 1) > 0:
                rho = i
                
        theta = (1 - cssv[rho]) / (rho + 1)
        return np.maximum(v + theta, 0)

# WASC strategy
class WASC(BaseStrategy):
    """Weakly-aggregated specialized CRP strategy."""
    
    def __init__(self, gamma=0.0025, u=1, epsilon=0.15):
        super().__init__("WASC")
        self.gamma = gamma  # Transaction cost rate
        self.u = u  # Learning rate
        self.epsilon = epsilon  # Active expert ratio
        
        # Initialize member variables
        self.experts_b = None  # Fixed portfolio per expert (one-hot)
        self.G = None  # Cumulative return per expert
        self.adjusted_b_prev_experts = None  # Previous adjusted portfolio per expert
        self.awake_experts = None  # Active experts
        self.prev_adjusted_b = None  # Previous adjusted portfolio
    
    def allocate(self, data, current_weights=None):
        """Allocate weights using WASC."""
        n_assets = data.shape[1]
        
        # Initialize on first call
        if current_weights is None or self.experts_b is None:
            current_weights = np.ones(n_assets) / n_assets
            self.experts_b = np.eye(n_assets)  # Fixed portfolio per expert (one-hot)
            self.G = np.zeros(n_assets)  # Cumulative return per expert
            self.adjusted_b_prev_experts = np.eye(n_assets)  # Previous adjusted portfolio
            self.awake_experts = np.arange(n_assets)  # All experts active initially
            self.prev_adjusted_b = np.zeros(n_assets)  # Previous adjusted portfolio
            return current_weights
        
        # Convert dict weights to array if needed
        if isinstance(current_weights, dict):
            weight_array = np.zeros(n_assets)
            for i, key in enumerate(data.columns):
                weight_array[i] = current_weights.get(key, 0)
            current_weights = weight_array
            
        # Get current period price relatives
        if len(data) < 2:
            return current_weights
            
        price_relatives = data.iloc[-1] / data.iloc[-2]
        price_relatives = price_relatives.values
        
        # Compute portfolio return and transaction cost
        portfolio_return = np.dot(current_weights, price_relatives)
        if portfolio_return <= 0:
            portfolio_return = 1e-10  # Avoid divide-by-zero
        
        # Transaction cost calculation
        distance = np.sum(np.abs(current_weights - self.prev_adjusted_b))
        transaction_cost = 1 - self.gamma * distance
        
        # Update adjusted portfolio
        adjusted_b = (current_weights * price_relatives) / portfolio_return
        
        # Compute k for each expert (number of active experts)
        k = max(1, int(self.epsilon * n_assets))
        
        # Update each expert's cumulative return and adjusted portfolio
        t = min(len(data), 100)  # Cap t to avoid numerical issues
        
        for n in range(n_assets):
            if n in self.awake_experts:
                # Active expert: use its own portfolio
                expert_b = self.experts_b[n]
                expert_return = np.dot(expert_b, price_relatives)
                prev_adj = self.adjusted_b_prev_experts[n]
                cost = 1 - self.gamma * np.sum(np.abs(expert_b - prev_adj))
                g_tn = np.log(expert_return * cost + 1e-10)
                new_adj = (expert_b * price_relatives) / (expert_return + 1e-10)
            else:
                # Inactive expert: use WASC portfolio
                expert_return = portfolio_return
                prev_adj = self.adjusted_b_prev_experts[n]
                cost = 1 - self.gamma * np.sum(np.abs(current_weights - prev_adj))
                g_tn = np.log(expert_return * cost + 1e-10)
                new_adj = adjusted_b
            
            self.G[n] += g_tn
            self.adjusted_b_prev_experts[n] = new_adj
        
        # Compute distance between adjusted portfolio and each expert
        distances = np.array([np.sum(np.abs(adjusted_b - self.experts_b[n])) for n in range(n_assets)])
        
        # Select active experts for next period
        sorted_indices = np.argsort(distances)
        self.awake_experts = sorted_indices[:k]
        
        # Compute weights of active experts
        weights = np.exp(self.u * self.G[self.awake_experts] / np.sqrt(t))
        sum_weights = np.sum(weights)
        
        if sum_weights == 0:
            p = np.ones(len(self.awake_experts)) / len(self.awake_experts)
        else:
            p = weights / sum_weights
        
        # Update next-period portfolio
        next_b = np.zeros(n_assets)
        for i, idx in enumerate(self.awake_experts):
            next_b += p[i] * self.experts_b[idx]
        
        # Normalize to sum to 1
        if np.sum(next_b) > 0:
            next_b /= np.sum(next_b)
        else:
            next_b = np.ones(n_assets) / n_assets
        
        # Update tracking variables
        self.prev_adjusted_b = adjusted_b.copy()
        
        return next_b

# CAEGc strategy
class CAEGc(BaseStrategy):
    """Continuous aggregation EG strategy (with transaction costs)."""
    
    def __init__(self, eta=0.05, gamma_tc=0.01):
        super().__init__("CAEGc")
        self.eta = eta  # Learning rate
        self.gamma_tc = gamma_tc  # Transaction cost coefficient
    
    def allocate(self, data, current_weights=None):
        """Allocate weights using CAEGc (with transaction costs)."""
        n_assets = data.shape[1]
        
        if current_weights is None or len(data) < 2:
            return np.ones(n_assets) / n_assets
        
        # Convert dict weights to array if needed
        if isinstance(current_weights, dict):
            weight_array = np.zeros(n_assets)
            for i, key in enumerate(data.columns):
                weight_array[i] = current_weights.get(key, 0)
            current_weights = weight_array
            
        # Get recent price relatives
        price_relatives = data.iloc[-1] / data.iloc[-2]
        
        if isinstance(price_relatives, pd.Series):
            price_relatives = price_relatives.values
        
        # EG core update: exponentiated gradient
        numerator = current_weights * np.exp(self.eta * price_relatives)
        eg_weights = numerator / np.sum(numerator)
        
        # Adjust for transaction costs by penalizing deviations
        delta = eg_weights - current_weights
        
        # L1 norm penalty
        l1_penalty = np.sum(np.abs(delta)) * self.gamma_tc
        
        # Adjusted new weights
        new_weights = current_weights + delta * (1 - l1_penalty)
        
        # Ensure weights are valid
        if np.sum(new_weights) != 1 or np.any(new_weights < 0):
            new_weights = self._simplex_projection(new_weights)
        
        return new_weights
    
    def _simplex_projection(self, v):
        """Project vector onto simplex."""
        n = len(v)
        if np.sum(v) == 1 and np.all(v >= 0):
            return v
            
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        
        rho = 0
        for i in range(n):
            if u[i] + (1 - cssv[i]) / (i + 1) > 0:
                rho = i
                
        theta = (1 - cssv[rho]) / (rho + 1)
        return np.maximum(v + theta, 0)

# RLCVAR strategy (CVaR-based deep RL portfolio optimization)
class RLCVAR(BaseStrategy):
    """Portfolio strategy based on CVaR and deep reinforcement learning."""
    
    def __init__(self, window_size=30, beta=0.95, use_cvar=True):
        """Initialize RLCVAR strategy."""
        super().__init__("RLCVAR")
        self.window_size = window_size
        self.beta = beta
        self.use_cvar = use_cvar
        self.device = torch.device('cuda:0' if torch.cuda.is_avaSIMble() else 'cpu')
        self.model = None
        self.historical_data = []  # Store historical data for features
        self.initialized = False   # Model initialization flag
        
    def allocate(self, data, current_weights=None):
        """Allocate portfolio weights for the current market state."""
        n_assets = data.shape[1]
        
        # Initialize model on first call
        if not self.initialized:
            self.init_model(n_assets)
            self.initialized = True
        
        # Handle insufficient data
        if len(data) < self.window_size:
            if current_weights is not None:
                # Convert dict weights to array if needed
                if isinstance(current_weights, dict):
                    weight_array = np.zeros(n_assets)
                    for i, key in enumerate(data.columns):
                        weight_array[i] = current_weights.get(key, 0)
                    return weight_array
                return current_weights
            else:
                return np.ones(n_assets) / n_assets
        
        # Extract market features (OHLC)
        features = self.extract_features(data)
        
        # Predict weights with model
        try:
            with torch.no_grad():
                # Flatten features to match model input
                flattened_features = features.reshape(1, -1)
                state_tensor = torch.FloatTensor(flattened_features).to(self.device)
                weights = self.model(state_tensor).squeeze(0).cpu().numpy()
        except Exception as e:
            print(f"RLCVAR prediction error: {str(e)}")
            print(f"Feature shape: {features.shape}, flattened: {features.reshape(1, -1).shape}")
            # Fallback to equal weights
            weights = np.ones(n_assets) / n_assets
            return weights
        
        # Ensure weights sum to 1
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            weights = np.ones(n_assets) / n_assets
            
        return weights
    
    def init_model(self, n_assets):
        """Initialize the model."""
        # Create feature input size
        input_dim = self.window_size * 4  # 4 features per asset (OHLC)
        
        # Create policy network
        self.model = PolicyNetwork(input_dim, n_assets).to(self.device)
        
        # Try loading pretrained weights (if available)
        try:
            self.model.load_state_dict(torch.load('models/rl_cvar_model.pth', map_location=self.device))
            print("Loaded pretrained RLCVAR model")
        except:
            print("No pretrained RLCVAR model found; using random init")
    
    def extract_features(self, data):
        """Extract features required by the RLCVAR model."""
        # Get recent window_size data
        recent_data = data.iloc[-self.window_size:].copy()
        
        # Approximate OHLC from close-only data
        n_assets = recent_data.shape[1]
        
        # Create feature vector [window_size * 4] to match model input
        features = np.zeros((self.window_size * 4))
        
        # Use the first asset to build features
        asset_idx = 0
        
        for i, (_, row) in enumerate(recent_data.iterrows()):
            if i >= self.window_size:
                break
                
            # Approximate OHLC: open=prev close*0.995, high=close*1.01, low=close*0.99
            open_price = row.iloc[asset_idx] * 0.995 if i > 0 else row.iloc[asset_idx]
            high_price = row.iloc[asset_idx] * 1.01
            low_price = row.iloc[asset_idx] * 0.99
            close_price = row.iloc[asset_idx]
            
            # Normalize prices relative to open
            if open_price > 0:  # Avoid divide-by-zero
                norm_open = 1.0  # Open relative to itself is 1
                norm_high = high_price / open_price
                norm_low = low_price / open_price
                norm_close = close_price / open_price
            else:
                norm_open = norm_high = norm_low = norm_close = 1.0
            
            # Fill feature vector in time order
            base_idx = i * 4
            features[base_idx] = norm_open
            features[base_idx+1] = norm_high
            features[base_idx+2] = norm_low
            features[base_idx+3] = norm_close
        
        return features

# Residual block (RLCVAR)
class ResidualBlock(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, in_dim)
        )
        
    def forward(self, x):
        return x + self.fc(x)

# Policy network (RLCVAR)
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, num_assets):
        super().__init__()
        hidden_dim = 256  # Hidden dimension
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            nn.Linear(hidden_dim, num_assets)
        )
        
    def forward(self, state):
        # state shape: (batch, window_size*4)
        logits = self.net(state)
        weights = torch.softmax(logits, dim=-1)  # Ensure weights sum to 1
        return weights

# DNA-S model components
class DifferentialSharpeRatio:
    """Differential Sharpe ratio calculator (Moody 1998)."""
    
    def __init__(self, eta=0.9):
        """Initialize the DSR calculator."""
        self.eta = eta
        self.reset()
        
    def reset(self):
        """Reset historical statistics."""
        self.A = 0.0  # First-order moment
        self.B = 0.0  # Second-order moment
        self.prev_ratio = 0.0  # Previous Sharpe ratio
        
    def __call__(self, current_return, risk_free_rate=0.0):
        """Compute differential Sharpe ratio reward."""
        # Excess return
        R_t = current_return - risk_free_rate
        
        # Update recursive statistics (Eq. 26a/26b)
        delta_A = (1 - self.eta) * (R_t - self.A)
        delta_B = (1 - self.eta) * (R_t**2 - self.B)
        
        # Compute differential Sharpe ratio (Eq. 26)
        denominator = np.sqrt(self.B - self.A**2 + 1e-8)
        S_t = self.A / denominator
        delta_S = (delta_A * (self.B - self.A**2) - 0.5 * self.A * delta_B) / ((self.B - self.A**2)**1.5 + 1e-8)
        DSR = delta_S * (1 - self.eta + self.eta * self.prev_ratio)
        
        # Update state variables
        self.A += delta_A
        self.B += delta_B
        self.prev_ratio = DSR
        
        return DSR

class DynamicAssetNetwork(nn.Module):
    """Dynamic asset network for DNA-S."""
    
    def __init__(self, input_size=7, gru_hidden_size=10, num_gru_layers=2):
        super().__init__()
        # GRU processes per-asset features
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=gru_hidden_size,
            num_layers=num_gru_layers,
            batch_first=True
        )
        
        # Portfolio weight generation layer
        self.fc = nn.Sequential(
            nn.Linear(gru_hidden_size + 1, 32),  # +1 for portfolio weight
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(gru_hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, asset_features, portfolio_weights):
        """Forward pass."""
        try:
            batch_size, num_assets = asset_features.shape[:2]
            feature_dim = asset_features.shape[2]
            
            # Reshape into per-asset sequences
            asset_features = asset_features.view(batch_size * num_assets, 1, feature_dim)
            
            # Process features through GRU
            _, hidden = self.gru(asset_features)  # Output [num_layers, batch*assets, hidden]
            hidden = hidden[-1]  # Last layer [batch*assets, hidden]
            hidden = hidden.view(batch_size, num_assets, -1)  # Reshape [batch, assets, hidden]
            
            # Concatenate portfolio weights
            combined = torch.cat([
                hidden,
                portfolio_weights.unsqueeze(-1)
            ], dim=-1)  # [batch, assets, hidden+1]
            
            # Generate action probabilities
            logits = self.fc(combined).squeeze(-1)  # [batch, assets]
            action_probs = torch.softmax(logits, dim=-1)
            
            # State value estimation
            values = self.value_net(hidden.mean(dim=1))  # [batch, 1]
            
            return action_probs, values
            
        except Exception as e:
            print(f"DynamicAssetNetwork forward error: {str(e)}")
            # Fallback to uniform weights
            uniform_weights = torch.ones_like(portfolio_weights) / portfolio_weights.shape[-1]
            zero_value = torch.zeros((batch_size, 1), device=portfolio_weights.device)
            return uniform_weights, zero_value

# DNA-S strategy
class DNAS(BaseStrategy):
    """Dynamic number of assets strategy using DSR as reward."""
    
    def __init__(self, hist_len=60, feature_dim=7, eta=0.9):
        """Initialize DNA-S strategy."""
        super().__init__("DNA-S")
        self.hist_len = hist_len
        self.feature_dim = feature_dim
        self.eta = eta
        self.device = torch.device('cuda:0' if torch.cuda.is_avaSIMble() else 'cpu')
        self.model = None
        self.dsr_calculator = DifferentialSharpeRatio(eta=eta)
        self.portfolio_value = 1.0  # Initial portfolio value
        self.prev_weights = None
        self.initialized = False
        
    def allocate(self, data, current_weights=None):
        """Allocate asset weights."""
        n_assets = data.shape[1]
        
        # Initialize model on first call
        if not self.initialized:
            self.init_model(n_assets)
            self.initialized = True
        
        # Handle insufficient data
        if len(data) < self.hist_len:
            if current_weights is not None:
                return current_weights
            else:
                return np.ones(n_assets) / n_assets
        
        # Extract market features
        features = self.extract_features(data)
        
        # Prepare current weights (uniform if missing)
        if current_weights is None:
            current_weights = np.ones(n_assets) / n_assets
        if self.prev_weights is None:
            self.prev_weights = current_weights
        
        # If model is available, predict weights
        if self.model is not None:
            try:
                with torch.no_grad():
                    # Reshape tensors to match model input
                    features_tensor = torch.FloatTensor(features).to(self.device)  # [n_assets*hist_len*feature_dim]
                    # Reshape to 3D with batch dimension
                    features_tensor = features_tensor.reshape(1, n_assets, -1)  # [1, n_assets, hist_len*feature_dim]
                    
                    weights_tensor = torch.FloatTensor(current_weights).unsqueeze(0).to(self.device)  # [1, n_assets]
                    
                    # Model prediction
                    new_weights, _ = self.model(features_tensor, weights_tensor)
                    new_weights = new_weights.squeeze(0).cpu().numpy()
            except Exception as e:
                print(f"DNA-S prediction error: {str(e)}")
                print(f"Feature shape: {features.shape}, reshaped: {features.reshape(1, n_assets, -1).shape}")
                # Fallback to uniform weights
                new_weights = np.ones(n_assets) / n_assets
        else:
            # If model unavailable, use uniform weights
            new_weights = np.ones(n_assets) / n_assets
        
        # Compute return (if previous weights exist)
        if hasattr(self, 'prev_weights') and self.prev_weights is not None:
            if len(data) > 1:
                # Get price relatives
                price_relatives = data.iloc[-1] / data.iloc[-2]
                
                # Compute return under previous weights
                portfolio_return = np.sum(self.prev_weights * price_relatives.values)
                
                # Update portfolio value
                new_portfolio_value = self.portfolio_value * portfolio_return
                
                # Compute DSR reward (for logging/diagnostics)
                dsr_reward = self.dsr_calculator(portfolio_return - 1.0)  # Convert to return
                
                # Update portfolio value
                self.portfolio_value = new_portfolio_value
        
        # Update previous weights
        self.prev_weights = new_weights
        
        # Ensure weights sum to 1
        if np.sum(new_weights) > 0:
            new_weights = new_weights / np.sum(new_weights)
        else:
            new_weights = np.ones(n_assets) / n_assets
            
        return new_weights
    
    def init_model(self, n_assets):
        """Initialize model."""
        # Create model
        self.model = DynamicAssetNetwork(
            input_size=self.feature_dim * self.hist_len,  # Flattened feature dimension
            gru_hidden_size=10,
            num_gru_layers=2
        ).to(self.device)
        
        # Try loading pretrained weights (if available)
        try:
            self.model.load_state_dict(torch.load('models/dna_s_model.pth', map_location=self.device))
            print("Loaded pretrained DNA-S model")
        except:
            print("No pretrained DNA-S model found; using random init")
            
        # Initialize DSR calculator
        self.dsr_calculator = DifferentialSharpeRatio(eta=self.eta)
    
    def extract_features(self, data):
        """Extract asset features."""
        # Get last hist_len days
        recent_data = data.iloc[-self.hist_len:].copy()
        
        n_assets = recent_data.shape[1]
        
        # Initialize feature array [n_assets * hist_len * feature_dim] - flattened
        features = np.zeros((n_assets * self.hist_len * self.feature_dim))
        
        # Build features per asset
        for asset_idx in range(n_assets):
            price_series = recent_data.iloc[:, asset_idx].values
            
            for t in range(self.hist_len):
                # Compute basic features
                # 1. Price
                price = price_series[t]
                
                # 2. Short/long returns (if enough data)
                ret_1d = 0.0
                ret_5d = 0.0
                ret_10d = 0.0
                
                if t > 0:
                    ret_1d = price_series[t] / price_series[t-1] - 1.0
                if t >= 5:
                    ret_5d = price_series[t] / price_series[t-5] - 1.0
                if t >= 10:
                    ret_10d = price_series[t] / price_series[t-10] - 1.0
                
                # 3. Volatility estimate (if enough data)
                vol_5d = 0.0
                vol_10d = 0.0
                
                if t >= 5:
                    returns = np.diff(price_series[t-5:t+1]) / price_series[t-5:t]
                    vol_5d = np.std(returns)
                if t >= 10:
                    returns = np.diff(price_series[t-10:t+1]) / price_series[t-10:t]
                    vol_10d = np.std(returns)
                
                # 4. Price direction
                price_direction = 0.0
                if t > 0:
                    price_direction = 1.0 if price_series[t] > price_series[t-1] else -1.0
                
                # Fill feature array (flattened index)
                base_idx = (asset_idx * self.hist_len + t) * self.feature_dim
                features[base_idx] = price  # Price
                features[base_idx + 1] = ret_1d  # 1-day return
                features[base_idx + 2] = ret_5d  # 5-day return 
                features[base_idx + 3] = ret_10d  # 10-day return
                features[base_idx + 4] = vol_5d  # 5-day volatility
                features[base_idx + 5] = vol_10d  # 10-day volatility
                features[base_idx + 6] = price_direction  # Price direction
        
        # Normalize features
        mean = np.mean(features)
        std = np.std(features)
        if std > 0:
            features = (features - mean) / std
                    
        return features

# Strategy comparator
class StrategyComparator:
    """Portfolio strategy comparator."""
    
    def __init__(self, data_path, output_dir, start_date, end_date, config_path='config/default.yaml'):
        """Initialize the strategy comparator."""
        self.data_path = data_path
        self.output_dir = output_dir
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.config_path = config_path
        
        # Load config from path
        if not hasattr(self, 'config'):
            try:
                self.config = get_config(self.config_path)
                print(f"Loaded config from {self.config_path}")
            except FileNotFoundError:
                # Try adding LIMPPO_CNN prefix
                alt_config_path = os.path.join('LIMPPO_CNN', config_path)
                try:
                    self.config = get_config(alt_config_path)
                    self.config_path = alt_config_path
                    print(f"Loaded config from {self.config_path}")
                except FileNotFoundError:
                    print(f"Config file not found: {config_path} or {alt_config_path}")
                    raise
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize strategy set - add new strategies
        self.strategies = [
            WASC(gamma=0.0025, u=1, epsilon=0.15),  # Weakly-aggregated specialized CRP
            CAEGc(eta=0.05, gamma_tc=0.01),         # Continuous aggregation EG (with TC)
            RLCVAR(window_size=30, beta=0.95),      # CVaR-based deep RL strategy
            DNAS(hist_len=60, feature_dim=7, eta=0.9),  # Dynamic number of assets with DSR
            # Other strategies commented out; keep only the above
            # UBAH(),           # Buy and hold
            # UCRP(),           # Uniform constant rebalanced
            # EG(),             # Exponentiated gradient
            # Anticor(),        # Anti-correlation
            # PAMR(),           # Passive aggressive mean reversion
            # OLMAR(),          # Online moving average reversion
            # CORN_K(),         # Correlation-driven KNN
            # CWMR_Var(),       # Confidence-weighted mean reversion (var)
            # RMR(),            # Robust median reversion
            # TCO1()            # Transaction cost optimization
        ]
        
        # Load data
        self.load_data()
    
    def load_data(self):
        """Load stock data using main experiment settings."""
        try:
            # Update date range in config
            self.config['data']['start_date'] = self.start_date.strftime('%Y-%m-%d')
            self.config['data']['end_date'] = self.end_date.strftime('%Y-%m-%d')
            
            # If data path provided, update data directory
            if self.data_path:
                self.config['data']['data_dir'] = self.data_path
            
            # Create data loader via factory
            print("Creating data loader via factory...")
            data_loader = create_data_loader(self.config)
            
            # Load data
            print("Starting data load...")
            data_loader.load_data()
            
            # Convert data into format suitable for strategy comparison
            all_data = {}
            for symbol, data in data_loader.all_stock_data.items():
                if 'close' in data.columns:
                    all_data[symbol] = data['close']
                elif 'Close' in data.columns:
                    all_data[symbol] = data['Close']
                elif 'raw_Close' in data.columns:
                    all_data[symbol] = data['raw_Close']
            
            if not all_data:
                raise ValueError("No valid close price data found")
            
            # Combine all stock data into one DataFrame
            self.data = pd.DataFrame(all_data)
            
            # Ensure index is datetime
            if not isinstance(self.data.index, pd.DatetimeIndex):
                if 'date' in data_loader.all_stock_data[list(data_loader.all_stock_data.keys())[0]].columns:
                    dates = data_loader.all_stock_data[list(data_loader.all_stock_data.keys())[0]]['date']
                    self.data.index = pd.to_datetime(dates)
            
            # Filter date range
            self.data = self.data[(self.data.index >= self.start_date) & 
                                (self.data.index <= self.end_date)]
            
            # Check and handle NaNs
            nan_count = self.data.isna().sum().sum()
            if nan_count > 0:
                print(f"Found {nan_count} NaNs, preprocessing...")
                
                # Check per-stock NaN counts
                stock_nan_counts = self.data.isna().sum()
                stocks_with_nans = stock_nan_counts[stock_nan_counts > 0]
                if len(stocks_with_nans) > 0:
                    print("Stocks with NaNs:")
                    for symbol, count in stocks_with_nans.items():
                        print(f"  {symbol}: {count} NaNs ({count/len(self.data)*100:.2f}%)")
                
                # Check daily NaN counts
                daily_nan_counts = self.data.isna().sum(axis=1)
                days_with_nans = daily_nan_counts[daily_nan_counts > 0]
                if len(days_with_nans) > 0:
                    print(f"{len(days_with_nans)} days have NaNs; top 5 by count:")
                    for day, count in days_with_nans.nlargest(5).items():
                        print(f"  {day.strftime('%Y-%m-%d')}: {count} NaNs ({count/self.data.shape[1]*100:.2f}%)")
                
                # Preprocess NaNs with filling
                print("Filling NaNs with forward/backward fill...")
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
            raise ValueError(f"Failed to load stock data: {str(e)}")
    
    def run_backtest(self, initial_capital=1000000):
        """Run backtest."""
        results = {}
        
        for strategy in self.strategies:
            print(f"Backtesting strategy {strategy.name}...")
            
            # Initialize capital and weights
            portfolio_value = initial_capital
            weights = None
            portfolio_values = [portfolio_value]
            dates = [self.data.index[0]]
            
            # Track turnover
            previous_weights = None
            weights_changes = []
            
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
                            # Record previous weights
                            if weights is not None:
                                previous_weights = weights.copy()
                            
                            # Compute new portfolio weights
                            weights = strategy.allocate(hist_data, weights)
                            
                            # Ensure weights are numpy array
                            if isinstance(weights, dict):
                                # Convert dict to numpy array
                                weight_array = np.zeros(len(self.data.columns))
                                for j, symbol in enumerate(self.data.columns):
                                    weight_array[j] = weights.get(symbol, 0)
                                weights = weight_array
                                
                            # Ensure weight dimension matches asset count
                            if len(weights) != len(self.data.columns):
                                print(f"Warning: weight dim ({len(weights)}) != asset count ({len(self.data.columns)})")
                                # Use uniform weights
                                weights = np.ones(len(self.data.columns)) / len(self.data.columns)
                            
                            # Validate weights
                            if np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
                                print(f"Warning: {strategy.name} produced NaN weights; using uniform")
                                weights = np.ones(len(self.data.columns)) / len(self.data.columns)
                            
                            # Record weight changes
                            if previous_weights is not None and len(previous_weights) == len(weights):
                                weight_changes = weights - previous_weights
                                weights_changes.append(weight_changes)
                            
                        except Exception as e:
                            print(f"Error: {strategy.name} weight allocation failed: {str(e)}")
                            # Use uniform weights
                            weights = np.ones(len(self.data.columns)) / len(self.data.columns)
                    
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
                    # Replace NaN/inf with previous valid value
                    valid_values = ~(np.isnan(portfolio_values_array) | np.isinf(portfolio_values_array))
                    last_valid_idx = np.where(valid_values)[0][0]
                    last_valid_value = portfolio_values_array[last_valid_idx]
                    
                    for i in range(len(portfolio_values_array)):
                        if np.isnan(portfolio_values_array[i]) or np.isinf(portfolio_values_array[i]):
                            portfolio_values_array[i] = last_valid_value
                        else:
                            last_valid_value = portfolio_values_array[i]
                
                # Compute performance metrics
                metrics = calculate_metrics(portfolio_values_array, weights_changes)
                
                # Store results
                results[strategy.name] = {
                    'portfolio_values': portfolio_values,
                    'dates': dates,
                    'metrics': metrics,
                    'weights_changes': weights_changes
                }
                
                print(f"{strategy.name} results:")
                print(f"  Total return: {metrics['total_ret']:.4f}")
                print(f"  Annual return: {metrics['annual_ret']:.4f}")
                print(f"  Sharpe ratio: {metrics['sharpe_rat']:.4f}")
                print(f"  Max drawdown: {metrics['max_drawd']:.4f}")
                print(f"  Win rate: {metrics['win_rate']:.4f}")
                print(f"  Volatility: {metrics['volatility']:.4f}")
                print(f"  Sortino ratio: {metrics['sortino_r']:.4f}")
                print(f"  Turnover: {metrics['turnover_r']:.4f}")
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
                    'turnover_r': 0.0
                }
                # Store empty result
                results[strategy.name] = {
                    'portfolio_values': [initial_capital],
                    'dates': [self.data.index[0]],
                    'metrics': metrics,
                    'error': str(e)
                }
        
        return results
    
    def visualize_results(self, results):
        """Visualize backtest results."""
        # Set font style
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 12
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Plot portfolio value comparison
        try:
            plt.figure(figsize=(12, 8))
            
            for strategy_name, result in results.items():
                plt.plot(result['dates'], 
                        np.array(result['portfolio_values']) / result['portfolio_values'][0], 
                        label=strategy_name)
            
            plt.title('Portfolio Value Comparison (Normalized)', fontfamily='Times New Roman', fontsize=14)
            plt.xlabel('Date', fontfamily='Times New Roman')
            plt.ylabel('Normalized Value', fontfamily='Times New Roman')
            plt.legend(prop={'family': 'Times New Roman'})
            plt.grid(True)
            
            # Save figure
            plt.savefig(os.path.join(self.output_dir, 'portfolio_comparison.png'), dpi=300)
            plt.close()
        except Exception as e:
            print(f"Error plotting portfolio value comparison: {str(e)}")
        
        # Plot performance metrics comparison
        metrics = ['total_ret', 'annual_ret', 'sharpe_rat', 
                  'max_drawd', 'win_rate', 'volatility', 
                  'sortino_r', 'turnover_r']
        
        metric_names = {
            'total_ret': 'Total Return',
            'annual_ret': 'Annual Return',
            'sharpe_rat': 'Sharpe Ratio',
            'max_drawd': 'Maximum Drawdown',
            'win_rate': 'Win Rate',
            'volatility': 'Volatility',
            'sortino_r': 'Sortino Ratio',
            'turnover_r': 'Turnover Rate'
        }
        
        for metric in metrics:
            try:
                plt.figure(figsize=(10, 6))
                
                values = [result['metrics'][metric] for result in results.values()]
                strategies = list(results.keys())
                
                bars = plt.bar(strategies, values)
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.4f}',
                            ha='center', va='bottom', rotation=0,
                            fontfamily='Times New Roman')
                
                plt.title(f'{metric_names[metric]} Comparison', fontfamily='Times New Roman', fontsize=14)
                plt.xlabel('Strategy', fontfamily='Times New Roman')
                plt.ylabel(metric_names[metric], fontfamily='Times New Roman')
                plt.xticks(rotation=45, fontfamily='Times New Roman')
                plt.yticks(fontfamily='Times New Roman')
                plt.tight_layout()
                
                # Save figure
                plt.savefig(os.path.join(self.output_dir, f'{metric}_comparison.png'), dpi=300)
                plt.close()
            except Exception as e:
                print(f"Error plotting {metric} comparison: {str(e)}")
        
        # Save metrics table
        try:
            metrics_data = []
            for strategy_name, result in results.items():
                metrics_data.append({
                    'strategy': strategy_name,
                    **result['metrics']
                })
            
            metrics_df = pd.DataFrame(metrics_data)
            
            # Handle file permission issues
            try:
                metrics_df.to_csv(os.path.join(self.output_dir, 'metrics_comparison.csv'), index=False)
                print(f"Metrics comparison saved to {os.path.join(self.output_dir, 'metrics_comparison.csv')}")
            except PermissionError:
                # If permission error, use a different filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                alt_file_path = os.path.join(self.output_dir, f'metrics_comparison_{timestamp}.csv')
                metrics_df.to_csv(alt_file_path, index=False)
                print(f"Permission issue; metrics saved to fallback file: {alt_file_path}")
        except Exception as e:
            print(f"Error saving metrics table: {str(e)}")
        
        # Generate HTML report
        try:
            self._generate_html_report(results)
            print(f"HTML report generated: {os.path.join(self.output_dir, 'strategy_comparison_report.html')}")
        except Exception as e:
            print(f"Error generating HTML report: {str(e)}")
            # Try a simpler HTML report
            self._generate_simple_html_report(results)
    
    def _generate_simple_html_report(self, results):
        """Generate a simplified HTML report (fallback)."""
        try:
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Simple Strategy Comparison Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: right; }}
                    th {{ background-color: #f2f2f2; }}
                </style>
            </head>
            <body>
                <h1>Strategy Comparison Report ({self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')})</h1>
                <table>
                    <tr>
                        <th>Strategy</th>
                        <th>Total Return</th>
                        <th>Annual Return</th>
                        <th>Sharpe Ratio</th>
                        <th>Max Drawdown</th>
                        <th>Win Rate</th>
                        <th>Volatility</th>
                        <th>Sortino Ratio</th>
                        <th>Turnover Rate</th>
                    </tr>
            """
            
            for strategy_name, result in results.items():
                metrics = result['metrics']
                html_content += f"""
                    <tr>
                        <td>{strategy_name}</td>
                        <td>{metrics['total_ret']:.4f}</td>
                        <td>{metrics['annual_ret']:.4f}</td>
                        <td>{metrics['sharpe_rat']:.4f}</td>
                        <td>{metrics['max_drawd']:.4f}</td>
                        <td>{metrics['win_rate']:.4f}</td>
                        <td>{metrics['volatility']:.4f}</td>
                        <td>{metrics['sortino_r']:.4f}</td>
                        <td>{metrics['turnover_r']:.4f}</td>
                    </tr>
                """
            
            html_content += """
                </table>
                <p>Note: This is a simplified report. Some visuals may not be avaSIMble due to errors.</p>
            </body>
            </html>
            """
            
            # Save simplified HTML report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            simple_report_path = os.path.join(self.output_dir, f'simple_report_{timestamp}.html')
            with open(simple_report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            print(f"Simplified HTML report saved to: {simple_report_path}")
        except Exception as e:
            print(f"Simplified HTML report generation failed: {str(e)}")
    
    def save_results(self, results):
        """Save backtest results."""
        # Save portfolio value data
        for strategy_name, result in results.items():
            df = pd.DataFrame({
                'date': result['dates'],
                'portfolio_value': result['portfolio_values']
            })
            df.to_csv(os.path.join(self.output_dir, f'{strategy_name}_portfolio_values.csv'), index=False)
    
    def _generate_html_report(self, results):
        """Generate an HTML report."""
        try:
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Portfolio Strategy Comparison Report</title>
                <style>
                    body {{ font-family: 'Times New Roman', Times, serif; margin: 20px; }}
                    h1, h2 {{ color: #333366; }}
                    table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                    th, td {{ padding: 8px; text-align: right; border-bottom: 1px solid #ddd; }}
                    th {{ background-color: #f2f2f2; }}
                    tr:hover {{ background-color: #f5f5f5; }}
                    .strategy-name {{ text-align: left; font-weight: bold; }}
                    .chart-container {{ margin: 20px 0; }}
                </style>
            </head>
            <body>
                <h1>Portfolio Strategy Comparison Report</h1>
                <p>Backtest Period: {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}</p>
                
                <h2>Strategy Performance Metrics</h2>
                <table>
                    <tr>
                        <th>Strategy</th>
                        <th>Total Return</th>
                        <th>Annual Return</th>
                        <th>Sharpe Ratio</th>
                        <th>Max Drawdown</th>
                        <th>Win Rate</th>
                        <th>Volatility</th>
                        <th>Sortino Ratio</th>
                        <th>Turnover Rate</th>
                    </tr>
            """
            
            for strategy_name, result in results.items():
                metrics = result['metrics']
                html_content += f"""
                    <tr>
                        <td class="strategy-name">{strategy_name}</td>
                        <td>{metrics['total_ret']:.4f}</td>
                        <td>{metrics['annual_ret']:.4f}</td>
                        <td>{metrics['sharpe_rat']:.4f}</td>
                        <td>{metrics['max_drawd']:.4f}</td>
                        <td>{metrics['win_rate']:.4f}</td>
                        <td>{metrics['volatility']:.4f}</td>
                        <td>{metrics['sortino_r']:.4f}</td>
                        <td>{metrics['turnover_r']:.4f}</td>
                    </tr>
                """
            
            html_content += """
                </table>
                
                <div class="chart-container">
                    <h2>Portfolio Value Comparison</h2>
                    <img src="portfolio_comparison.png" width="100%" />
                </div>
                
                <h2>Performance Metrics Comparison</h2>
            """
            
            metrics = ['total_ret', 'annual_ret', 'sharpe_rat', 
                      'max_drawd', 'win_rate', 'volatility',
                      'sortino_r', 'turnover_r']
            
            metric_names = {
                'total_ret': 'Total Return',
                'annual_ret': 'Annual Return',
                'sharpe_rat': 'Sharpe Ratio',
                'max_drawd': 'Maximum Drawdown',
                'win_rate': 'Win Rate',
                'volatility': 'Volatility',
                'sortino_r': 'Sortino Ratio',
                'turnover_r': 'Turnover Rate'
            }
            
            for metric in metrics:
                html_content += f"""
                    <div class="chart-container">
                        <h3>{metric_names[metric]} Comparison</h3>
                        <img src="{metric}_comparison.png" width="100%" />
                    </div>
                """
            
            html_content += """
            </body>
            </html>
            """
            
            # Save HTML report with permission-safe handling
            try:
                report_path = os.path.join(self.output_dir, 'strategy_comparison_report.html')
                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                return report_path
            except PermissionError:
                # If permission error, use a different filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                alt_report_path = os.path.join(self.output_dir, f'strategy_comparison_report_{timestamp}.html')
                with open(alt_report_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                return alt_report_path
        except Exception as e:
            print(f"HTML report generation failed: {str(e)}")
            return None

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Portfolio strategy comparison')
    parser.add_argument('--data_path', type=str, default='data',
                      help='Data path')
    parser.add_argument('--output_dir', type=str, default='strategy_comparison_results',
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
    
    # Load config
    config_path = args.config
    if not os.path.exists(config_path):
        # Try adding LIMPPO_CNN prefix
        alt_config_path = os.path.join('LIMPPO_CNN', config_path)
        if os.path.exists(alt_config_path):
            config_path = alt_config_path
            print(f"Using config file: {config_path}")
        else:
            print(f"Config file not found: {config_path} or {alt_config_path}")
            raise FileNotFoundError(f"Config not found: {config_path} or {alt_config_path}")
    
    # Use same config loading as main experiment
    config = get_config(config_path)
    
    # Create strategy comparator
    comparator = StrategyComparator(
        data_path=args.data_path,
        output_dir=args.output_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        config_path=config_path
    )
    
    # Run backtest
    results = comparator.run_backtest(initial_capital=args.initial_capital)
    
    # Visualize results
    comparator.visualize_results(results)
    
    # Save results
    comparator.save_results(results)
    
    print(f"Strategy comparison complete; results saved to: {args.output_dir}")

