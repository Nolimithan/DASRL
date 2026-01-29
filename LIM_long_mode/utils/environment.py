"""
Trading environment module.

Implements a Gym-based RL trading environment with state, action, and reward logic.
"""

import numpy as np
import pandas as pd
import gym
from gym import spaces
from sklearn.decomposition import PCA
import warnings
import os
from datetime import datetime

# Ignore specific runtime warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

class TradingEnvironment(gym.Env):
    """Trading environment for RL portfolio management."""
    
    def __init__(self, config, data_loader, lim_calculator):
        """Initialize trading environment."""
        super(TradingEnvironment, self).__init__()
        
        self.config = config
        self.data_loader = data_loader
        self.lim_calculator = lim_calculator
        
        # Read parameters from config
        self.portfolio_size = config['environment']['portfolio_size']
        self.initial_capital = config['environment']['initial_capital']
        self.commission_rate = config['environment']['commission_rate']
        
        # Stop-loss settings
        self.enable_stop_loss = config['environment'].get('enable_stop_loss', False)  # Disabled by default
        self.stop_loss_threshold = config['environment'].get('stop_loss_threshold', 0.1)  # Default 10%
        
        # Stock group type
        self.group_type = config['environment'].get('group_type', 'TOP')  # Default TOP group
        
        # Reward parameters
        self.reward_params = config['rl']['reward_params'] if 'rl' in config and 'reward_params' in config['rl'] else {
            'lambda_1': 0.5,  # Portfolio variance penalty
            'lambda_2': 0.5,  # Turnover penalty
            'lambda_3': 1.0   # Drawdown penalty
        }
        
        # Action space
        self.action_space = gym.spaces.Box(
            low=0, high=1,
            shape=(self.portfolio_size,),
            dtype=np.float32
        )
        
        # Dynamically compute state dimension
        self.state_dim = self._calculate_state_dim()
        
        # Observation space
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32
        )
        
        # Portfolio state
        self.portfolio_value = self.initial_capital
        self.cash = self.initial_capital
        self.max_portfolio_value = self.initial_capital
        self.current_weights = {}
        self.previous_weights = {}
        self.holdings = {}  # Empty holdings
        self.shares = {}    # Empty share counts
        self.current_stocks = []
        self.previous_portfolio_value = self.initial_capital
        
        # Statistics
        self.rebalance_count = 0
        self.stop_loss_count = 0
        self.current_drawdown = 0
        self.transaction_cost_total = 0
        
        # History
        self.history = []
        
        # Trading-related
        self.stop_loss_triggered = False
        self.today_trades = {}
        
        # PCA-related
        self.state_buffer = []  # Store historical state samples
        self.buffer_size = 50   # Buffer size
        self.pca_fitted = False # PCA fitted flag
        self.pca_model = None   # PCA model
        
        # ATR-related settings
        self.atr_window = config.get('environment', {}).get('atr_window', 14)  # ATR window size
        self.volatility_range = config.get('environment', {}).get('volatility_range', (0.5, 2.0))  # Volatility range
        self.use_trend_adjustment = config.get('environment', {}).get('trend_adjustment', True)  # Trend adjustment
        self.atr_history = []  # ATR history
        self.stop_loss_threshold_history = []  # Stop-loss threshold history
        self.original_stop_loss_threshold = self.stop_loss_threshold  # Original threshold
        
    def _calculate_state_dim(self):
        """Compute state vector dimension."""
        # Market features (benchmark return, volatility, momentum)
        market_feature_count = 3  # Actual count used
        
        # Per-stock features (return, volatility)
        stock_feature_count = 2  # Basic features
        
        # Portfolio features (value, drawdown, change)
        portfolio_feature_count = 5
        
        # Total dim = market + stock * count + portfolio
        total_dim = market_feature_count + (stock_feature_count * self.portfolio_size) + portfolio_feature_count
        
        print(f"Computed state dimension: {total_dim}")
        # Keep dimension aligned for NN usage
        return total_dim
    
    def reset(self, start_date=None, end_date=None):
        """Reset environment state and initialize portfolio."""
        # Use config dates if not provided
        if start_date is None:
            start_date = self.config['data']['start_date']
        if end_date is None:
            end_date = self.config['data']['end_date']
            
        # Store original date strings for logging
        self.start_date_str = start_date
        self.end_date_str = end_date
            
        # Load market and stock data
        print(f"Loading data from {start_date} to {end_date}...")
        
        # Set date range
        self.data_loader.set_date_range(start_date, end_date)
        
        # Load market data
        market_data = self.data_loader.benchmark
        
        if market_data is None or len(market_data) == 0:
            raise ValueError("Failed to load market data; check date range and data source")
            
        self.market_data = market_data
        
        # Get all stock data
        self.all_stock_data = self.data_loader.all_stock_data
        
        if not self.all_stock_data:
            raise ValueError("No stock data loaded; check tickers and data source")
            
        # Reset current step
        self.current_step = 0
        
        # Initialize portfolio
        self.portfolio_value = self.initial_capital
        self.cash = self.initial_capital
        self.max_portfolio_value = self.initial_capital
        self.current_weights = {}
        self.previous_weights = {}
        self.holdings = {}  # Empty holdings
        self.shares = {}    # Empty share counts
        self.current_stocks = []
        self.stop_loss_triggered = False
        self.previous_portfolio_value = self.initial_capital
        
        # Reset history
        self.history = []
        
        # Reset statistics
        self.rebalance_count = 0
        self.stop_loss_count = 0
        self.max_drawdown = 0
        self.current_drawdown = 0
        self.transaction_cost_total = 0
        
        # Reset PCA-related state
        self.state_buffer = []
        self.pca_fitted = False
        self.pca_model = None
        
        # Select initial portfolio stocks
        self._select_portfolio_stocks(self.group_type)
        
        # Get initial state
        initial_state = self._get_state()
        
        return initial_state
    
    def step(self, action):
        """Execute a trading action."""
        # Safety check: stop if current_step reaches end
        if self.current_step >= len(self.market_data) - 1:
            print(f"Warning: current_step({self.current_step}) reached market_data end ({len(self.market_data)-1}); ending.")
            # Use last valid state
            last_state = self._get_state()
            return last_state, 0, True, {'portfolio_value': self.portfolio_value, 'day': 'Final'}
            
        # Save previous weights
        self.previous_weights = self.current_weights.copy() if self.current_weights else {}
        
        # Ensure action dimension matches portfolio size
        if len(action) != len(self.current_stocks):
            raise ValueError(f"Action dim ({len(action)}) != portfolio size ({len(self.current_stocks)})")
            
        # Normalize weights to sum to 1
        if np.sum(action) > 0:
            action = action / np.sum(action)
        else:
            # If all weights are <= 0, use uniform weights
            action = np.ones(len(action)) / len(action)
            
        # Map tickers to weights
        target_weights = dict(zip(self.current_stocks, action))
        
        # Apply trading constraints
        target_weights = self._apply_trading_constraints(target_weights)
        
        # Execute trades
        transaction_cost = self._execute_trades(target_weights)
        
        # Record current weights
        self.current_weights = target_weights
        
        # Ensure cash/portfolio state is valid
        self._ensure_valid_state()
        
        # Update portfolio value
        self._update_portfolio_value()
        
        # Check stop-loss
        stop_loss_triggered = self._check_stop_loss()
        
        # Check if rebalance is needed (select new stocks)
        rebalance_triggered = self._check_rebalance_trigger()
        
        # Only reselect stocks when rebalancing
        if rebalance_triggered:
            self._select_portfolio_stocks(self.group_type)
        
        # Compute reward for this step
        reward = self._calculate_reward()
        
        # Determine episode termination
        done = self.current_step >= len(self.market_data) - 2  # End one step early to avoid overflow
        
        # Get current date info (safe handling)
        try:
            # Ensure market_data is valid DataFrame
            if isinstance(self.market_data, pd.DataFrame) and 'date' in self.market_data.columns:
                # Ensure current_step is valid index
                if 0 <= self.current_step < len(self.market_data):
                    current_day = self.market_data.iloc[self.current_step]['date']
                else:
                    current_day = f"Step_{self.current_step}"
            else:
                current_day = f"Step_{self.current_step}"
        except Exception as e:
            print(f"Error getting current date info: {e}")
            current_day = f"Step_{self.current_step}"
        
        # Return info dict
        info = {
            'portfolio_value': self.portfolio_value,
            'weights': self.current_weights,
            'transaction_cost': transaction_cost,
            'stop_loss_triggered': stop_loss_triggered,
            'rebalance_triggered': rebalance_triggered,
            'current_stocks': self.current_stocks,
            'day': current_day
        }
        
        # Record state
        self._record_state(info)
        
        # Update portfolio date (after recording state)
        self.current_step += 1
        
        # Get next state
        next_state = self._get_state()
        
        # Ensure environment state is valid
        self._ensure_valid_state()
        
        return next_state, reward, done, info
    
    def _get_state(self):
        """Build state vector from market, stock, and portfolio features."""
        # Safety check: ensure current_step within range
        if self.current_step >= len(self.market_data):
            # If out of range, use last valid index
            safe_step = len(self.market_data) - 1
            print(f"Warning: current_step({self.current_step}) out of range ({len(self.market_data)}), using {safe_step}")
            current_date = self.market_data.iloc[safe_step]['date']
        else:
            # Get current date
            current_date = self.market_data.iloc[self.current_step]['date']
        
        # 1. Global market features
        market_features = self._get_market_features(current_date)
        
        # 2. Local stock features
        stock_features = self._get_stock_features(current_date)
        
        # 3. Portfolio features
        portfolio_features = self._get_portfolio_features()
        
        # Combine all features
        full_state = np.concatenate([market_features, stock_features, portfolio_features])
        
        # Simplified PCA logic for small sample sizes
        # If samples are insufficient, use Gaussian noise to create synthetic samples
        use_pca = self.config.get('features', {}).get('use_pca', False)
        if use_pca:
            # Read PCA components from config (default 20)
            n_components = self.config.get('features', {}).get('pca_components', 20)
            
            try:
                # Limit components to <= 80% of feature dimension
                feature_dim = len(full_state)
                max_allowed = max(1, int(feature_dim * 0.8))
                n_components = min(n_components, max_allowed)
                
                # Build synthetic dataset to ensure enough samples
                synthetic_samples = 50  # 50 synthetic samples
                
                # Use current sample as mean, add small Gaussian noise
                noise_scale = 0.001  # Small noise scale to preserve features
                synthetic_data = np.tile(full_state, (synthetic_samples, 1))
                synthetic_data += np.random.normal(0, noise_scale, synthetic_data.shape)
                
                # Apply PCA
                pca = PCA(n_components=n_components)
                state = pca.fit_transform([full_state])[0]
                
                # Log success and variance (first run or periodically)
                if self.current_step == 0 or self.current_step % 100 == 0:
                    explained_variance_ratio = pca.explained_variance_ratio_
                    cumulative_variance = np.cumsum(explained_variance_ratio)
                    print(f"PCA reduced: {feature_dim} -> {n_components}, "
                          f"variance explained: {cumulative_variance[-1]:.2f}")
                    
            except Exception as e:
                print(f"PCA failed: {e}, using original features")
                state = full_state
        else:
            state = full_state
        
        # Ensure state dimension matches observation space
        actual_dim = len(state)
        expected_dim = self.observation_space.shape[0]
        
        if actual_dim != expected_dim:
            # Truncate if too large; pad if too small
            if actual_dim > expected_dim:
                state = state[:expected_dim]
            else:
                padding = np.zeros(expected_dim - actual_dim, dtype=np.float32)
                state = np.concatenate([state, padding])
                
        return state
    
    def _select_portfolio_stocks(self, group_type='TOP'):
        """Select portfolio stocks based on LIM and group type."""
        # Get current date
        current_date = self.market_data.iloc[self.current_step]['date']
        
        # Get all available tickers
        avaSIMble_stocks = list(self.data_loader.all_stock_data.keys())
        
        # Filter out ETFs
        avaSIMble_stocks = [s for s in avaSIMble_stocks if not (s.startswith('159') or s.startswith('512'))]
        
        # Ensure enough stocks are available
        if len(avaSIMble_stocks) < self.portfolio_size:
            print(f"Warning: available stocks ({len(avaSIMble_stocks)}) < portfolio size ({self.portfolio_size})")
            # If insufficient stocks, use all available
            self.current_stocks = avaSIMble_stocks
            print(f"Using all {len(self.current_stocks)} available stocks")
        else:
            # Check LIM calculator availability
            if self.lim_calculator is not None:
                try:
                    # Select stocks by group type
                    if group_type == 'TOP':
                        # Use LIM to select top stocks
                        top_stocks = self.lim_calculator.get_top_stocks(
                            self.data_loader.all_stock_data,  # Use all_stock_data
                            self.data_loader.benchmark['daily_return'],  # Benchmark returns
                            current_date,  # Current date
                            top_n=self.portfolio_size
                        )
                        selected_stocks = top_stocks
                        print(f"Selected TOP group: {len(selected_stocks)} stocks")
                    elif group_type == 'MIDDLE' or group_type == 'LOW':
                        # Get all stocks sorted by LIM*
                        sorted_stocks = self.lim_calculator.get_sorted_stocks(
                            self.data_loader.all_stock_data,
                            self.data_loader.benchmark['daily_return'],
                            current_date
                        )
                        
                        total_stocks = len(sorted_stocks)
                        if total_stocks < self.portfolio_size * 3:
                            print(f"Warning: total stocks ({total_stocks}) insufficient for 3 groups")
                            # If insufficient, still use top stocks
                            selected_stocks = sorted_stocks[:self.portfolio_size]
                        else:
                            if group_type == 'MIDDLE':
                                # Select middle group stocks
                                middle_start = total_stocks // 3
                                selected_stocks = sorted_stocks[middle_start:middle_start+self.portfolio_size]
                                print(f"Selected MIDDLE group: {len(selected_stocks)} stocks")
                            else:  # 'LOW'
                                # Select bottom group stocks
                                selected_stocks = sorted_stocks[-self.portfolio_size:]
                                print(f"Selected LOW group: {len(selected_stocks)} stocks")
                    else:
                        # Unknown group type, use top group
                        top_stocks = self.lim_calculator.get_top_stocks(
                            self.data_loader.all_stock_data,
                            self.data_loader.benchmark['daily_return'],
                            current_date,
                            top_n=self.portfolio_size
                        )
                        selected_stocks = top_stocks
                        print(f"Unknown group '{group_type}', defaulting to TOP: {len(selected_stocks)} stocks")
                    
                    # Randomly fill if not enough selected
                    if len(selected_stocks) < self.portfolio_size:
                        print(f"Selected stocks: {len(selected_stocks)}, required: {self.portfolio_size}")
                        
                        # Find unselected stocks
                        remaining_stocks = [s for s in avaSIMble_stocks if s not in selected_stocks]
                        
                        # Add needed stocks
                        additional_count = min(self.portfolio_size - len(selected_stocks), len(remaining_stocks))
                        if additional_count > 0:
                            print(f"Randomly added {additional_count} stocks")
                            additional_stocks = np.random.choice(
                                remaining_stocks, 
                                size=additional_count,
                                replace=False
                            ).tolist()
                            selected_stocks.extend(additional_stocks)
                    
                    self.current_stocks = selected_stocks
                except Exception as e:
                    print(f"LIM selection failed: {str(e)}")
                    print("Falling back to random selection")
                    # Random selection fallback
                    self.current_stocks = np.random.choice(
                        avaSIMble_stocks, 
                        size=min(self.portfolio_size, len(avaSIMble_stocks)),
                        replace=False
                    ).tolist()
            else:
                # If no LIM calculator, use random selection
                print("No LIM calculator provided, using random selection")
                self.current_stocks = np.random.choice(
                    avaSIMble_stocks, 
                    size=min(self.portfolio_size, len(avaSIMble_stocks)),
                    replace=False
                ).tolist()
        
        print(f"Selected {len(self.current_stocks)} stocks for the portfolio")
        
        # Update current holdings data
        self.current_stock_data = {}
        for symbol in self.current_stocks:
            if symbol in self.data_loader.all_stock_data:
                self.current_stock_data[symbol] = self.data_loader.all_stock_data[symbol]
        
        # Initialize equal weights
        equal_weight = 1.0 / len(self.current_stocks) if self.current_stocks else 0
        self.current_weights = {symbol: equal_weight for symbol in self.current_stocks}
    
    def _apply_trading_constraints(self, target_weights):
        """Apply trading constraints to limit daily turnover."""
        # If no current weights, return target
        if not self.previous_weights:
            return target_weights
            
        # Create a new weight dict
        constrained_weights = {}
        
        # Compute total weight change
        total_weight_change = 0
        for symbol in set(list(self.previous_weights.keys()) + list(target_weights.keys())):
            prev_weight = self.previous_weights.get(symbol, 0)
            target_weight = target_weights.get(symbol, 0)
            total_weight_change += abs(target_weight - prev_weight)
        
        # If total change exceeds max, scale proportionally
        max_allowed_change = 0.2  # Max daily turnover
        
        if total_weight_change > max_allowed_change:
            # Compute scale factor
            reduction_factor = max_allowed_change / total_weight_change
            
            # Apply scale factor
            for symbol in set(list(self.previous_weights.keys()) + list(target_weights.keys())):
                prev_weight = self.previous_weights.get(symbol, 0)
                target_weight = target_weights.get(symbol, 0)
                
                # Compute constrained weight
                weight_change = target_weight - prev_weight
                constrained_weight = prev_weight + weight_change * reduction_factor
                
                # Ensure non-negative weight
                constrained_weight = max(0, constrained_weight)
                
                # Save into result dict
                if constrained_weight > 0:
                    constrained_weights[symbol] = constrained_weight
        else:
            # If within limit, use target weights
            constrained_weights = target_weights.copy()
            
        # Ensure weights sum to 1
        total_weight = sum(constrained_weights.values())
        if total_weight > 0:
            for symbol in constrained_weights:
                constrained_weights[symbol] /= total_weight
                
        return constrained_weights
    
    def _execute_trades(self, target_weights):
        """Execute trades to reach target weights."""
        # Get current date and timestamp
        current_date = self.market_data.iloc[self.current_step]['date']
        current_timestamp = pd.Timestamp(current_date)
        
        # Initialize trade log
        trade_records = []
        
        # Initialize holdings if needed
        if not hasattr(self, 'holdings') or self.holdings is None:
            self.holdings = {}
            
        # Compute current net asset value
        total_value = self.portfolio_value
        
        # Get current prices
        current_prices = {}
        for symbol in set(list(self.current_stocks) + list(self.holdings.keys())):
            try:
                if symbol in self.all_stock_data and self.current_step < len(self.all_stock_data[symbol]):
                    current_prices[symbol] = self.all_stock_data[symbol].iloc[self.current_step]['close']
            except Exception as e:
                print(f"Error fetching price for {symbol}: {e}")
        
        # Initialize share counts (first time)
        if not hasattr(self, 'shares') or self.shares is None:
            self.shares = {}
            # Convert holding value to shares
            for symbol, value in self.holdings.items():
                if symbol in current_prices and current_prices[symbol] > 0:
                    # Convert value to integer shares
                    self.shares[symbol] = int(value / current_prices[symbol])
        
        # Compute target holding values
        target_values = {}
        for symbol, weight in target_weights.items():
            target_values[symbol] = total_value * weight
            
        # Compute required adjustments
        adjustments = {}
        for symbol in set(list(self.shares.keys()) + list(target_weights.keys())):
            current_shares = self.shares.get(symbol, 0)
            current_value = current_shares * current_prices.get(symbol, 0)
            target_value = target_values.get(symbol, 0)
            adjustments[symbol] = target_value - current_value
        
        # Sell stocks first (including those not in target portfolio)
        for symbol in list(self.shares.keys()):
            if symbol not in target_weights or target_weights[symbol] == 0:
                if symbol in current_prices and current_prices[symbol] > 0:
                    shares_to_sell = self.shares[symbol]
                    if shares_to_sell > 0:
                        sell_value = shares_to_sell * current_prices[symbol]
                        print(f"[{current_timestamp}] Liquidate {symbol}: {shares_to_sell} shares at {current_prices[symbol]:.2f}, value {sell_value:.2f}")
                        self.cash += sell_value * (1 - self.commission_rate)
                        del self.shares[symbol]
                        
                        trade_records.append({
                            'timestamp': current_timestamp,
                            'type': 'sell',
                            'symbol': symbol,
                            'shares': shares_to_sell,
                            'price': current_prices[symbol],
                            'value': sell_value,
                            'reason': 'portfolio_adjustment'
                        })
        
        # Split into sell and buy actions
        sell_adjustments = {symbol: adj for symbol, adj in adjustments.items() if adj < 0}
        buy_adjustments = {symbol: adj for symbol, adj in adjustments.items() if adj > 0}
        
        # Execute all sells
        total_sell = 0
        sell_cost = 0
        for symbol, adjustment in sell_adjustments.items():
            if symbol in current_prices and current_prices[symbol] > 0:
                shares_to_sell = int(abs(adjustment) / current_prices[symbol])
                current_shares = self.shares.get(symbol, 0)
                shares_to_sell = min(shares_to_sell, current_shares)
                
                if shares_to_sell > 0:
                    sell_value = shares_to_sell * current_prices[symbol]
                    total_sell += sell_value
                    
                    self.shares[symbol] = current_shares - shares_to_sell
                    if self.shares[symbol] <= 0:
                        del self.shares[symbol]
                    
                    print(f"[{current_timestamp}] Sell {symbol}: {shares_to_sell} shares at {current_prices[symbol]:.2f}, value {sell_value:.2f}")
                    
                    trade_records.append({
                        'timestamp': current_timestamp,
                        'type': 'sell',
                        'symbol': symbol,
                        'shares': shares_to_sell,
                        'price': current_prices[symbol],
                        'value': sell_value,
                        'reason': 'rebalance'
                    })
        
        # Compute sell transaction costs
        sell_cost = total_sell * self.commission_rate
        
        # Update available cash
        self.cash = self.cash + total_sell - sell_cost
        
        # Compute available cash (keep 5% buffer)
        avaSIMble_cash = self.cash * 0.95
        
        # Compute total desired buys
        total_buy_desired = 0
        shares_to_buy_dict = {}
        
        for symbol, adjustment in buy_adjustments.items():
            if symbol in current_prices and current_prices[symbol] > 0:
                target_shares = int(adjustment / current_prices[symbol])
                buy_value = target_shares * current_prices[symbol]
                total_buy_desired += buy_value
                shares_to_buy_dict[symbol] = target_shares
        
        # If total buy + cost exceeds cash, scale down buys
        estimated_buy_cost = total_buy_desired * self.commission_rate
        if total_buy_desired + estimated_buy_cost > avaSIMble_cash:
            reduction_ratio = avaSIMble_cash / (total_buy_desired + estimated_buy_cost)
            print(f"Warning: scaling buys, cash: {avaSIMble_cash:.2f}, desired: {total_buy_desired:.2f}, ratio: {reduction_ratio:.2%}")
            
            # Reduce all buy quantities proportionally
            for symbol in shares_to_buy_dict:
                shares_to_buy_dict[symbol] = max(1, int(shares_to_buy_dict[symbol] * reduction_ratio))
        
        # Execute all buys
        total_buy = 0
        buy_cost = 0
        
        for symbol, adjustment in buy_adjustments.items():
            if symbol in current_prices and current_prices[symbol] > 0:
                # Use precomputed (possibly scaled) share counts
                shares_to_buy = shares_to_buy_dict.get(symbol, 0)
                
                if shares_to_buy > 0:
                    # Check cash again
                    buy_value = shares_to_buy * current_prices[symbol]
                    buy_cost_estimate = buy_value * self.commission_rate
                    
                    # Ensure cash is sufficient
                    if buy_value + buy_cost_estimate <= self.cash:
                        total_buy += buy_value
                        
                        self.shares[symbol] = self.shares.get(symbol, 0) + shares_to_buy
                        
                        print(f"[{current_timestamp}] Buy {symbol}: {shares_to_buy} shares at {current_prices[symbol]:.2f}, value {buy_value:.2f}")
                        
                        trade_records.append({
                            'timestamp': current_timestamp,
                            'type': 'buy',
                            'symbol': symbol,
                            'shares': shares_to_buy,
                            'price': current_prices[symbol],
                            'value': buy_value,
                            'reason': 'rebalance'
                        })
                        
                        # Update cash immediately for next buy
                        self.cash -= buy_value
                        self.cash -= buy_value * self.commission_rate
                    else:
                        print(f"Warning: skip buy {symbol}, insufficient cash. Need {buy_value + buy_cost_estimate:.2f}, have {self.cash:.2f}")
        
        # Compute buy transaction costs
        buy_cost = total_buy * self.commission_rate
        
        # Update cash (avoid negative)
        self.cash = max(0, self.cash - buy_cost)
        
        # Update holdings value
        self.holdings = {}
        for symbol, shares in self.shares.items():
            if symbol in current_prices:
                self.holdings[symbol] = shares * current_prices[symbol]
        
        # Compute total transaction cost
        transaction_cost = sell_cost + buy_cost
        
        # Accumulate total transaction cost
        self.transaction_cost_total += transaction_cost
        
        # Update today's trade record
        self.today_trades = {
            'timestamp': current_timestamp,
            'trades': trade_records,
            'adjustments': adjustments,
            'transaction_cost': transaction_cost
        }
        
        # Compute total holdings value
        holdings_total_value = sum(self.holdings.values()) if self.holdings else 0.0
        
        # Compute portfolio value (holdings + cash)
        calculated_portfolio_value = holdings_total_value + self.cash
        
        # Ensure holdings and portfolio value are consistent
        # If discrepancy is large, log and update
        if abs(calculated_portfolio_value - self.portfolio_value) > 1.0:  # Allow <= 1 unit float error
            print(f"Warning: portfolio value mismatch, update: {self.portfolio_value:.2f} -> {calculated_portfolio_value:.2f}")
            print(f"Holdings value: {holdings_total_value:.2f}, cash: {self.cash:.2f}")
            self.portfolio_value = calculated_portfolio_value
        
        # If trades occurred, print summary
        if trade_records:
            print(f"\n[{current_timestamp}] Trade summary:")
            print(f"Total buy: {total_buy:.2f}")
            print(f"Total sell: {total_sell:.2f}")
            print(f"Transaction cost: {transaction_cost:.2f}")
            print(f"Cash: {self.cash:.2f}")
            print(f"Holdings value: {holdings_total_value:.2f}")
            print(f"Total value: {self.portfolio_value:.2f}\n")
        
        return transaction_cost
    
    def _update_portfolio_value(self):
        """Update portfolio value from market moves and costs."""
        # Deduct transaction costs
        self.portfolio_value -= self.transaction_cost_total
        self.transaction_cost_total = 0  # Reset to avoid double counting
        
        # Store previous value for returns
        old_portfolio_value = self.portfolio_value
        
        # If not last step, compute next-step returns
        if self.current_step + 1 < len(self.market_data):
            # Compute holdings return (excluding cash)
            portfolio_return = 0.0
            valid_weights_sum = 0.0  # Track valid weight sum
            
            # Compute total stock holdings value (non-cash)
            stock_holdings_value = 0.0
            if hasattr(self, 'holdings') and self.holdings:
                stock_holdings_value = sum(self.holdings.values())
            
            # Compute returns per stock
            for symbol in self.current_stocks:
                if symbol in self.current_weights:
                    weight = self.current_weights[symbol]
                    # Get current and next close
                    try:
                        if symbol in self.all_stock_data and self.current_step < len(self.all_stock_data[symbol]) and self.current_step + 1 < len(self.all_stock_data[symbol]):
                            current_price = self.all_stock_data[symbol].iloc[self.current_step]['close']
                            next_price = self.all_stock_data[symbol].iloc[self.current_step + 1]['close']
                            
                            # Validate prices
                            if np.isnan(current_price) or np.isnan(next_price) or current_price <= 0:
                                continue
                            
                            # Compute stock return
                            stock_return = (next_price / current_price) - 1
                            # Clip extreme values
                            stock_return = np.clip(stock_return, -0.1, 0.1)
                            
                            # Accumulate weighted return
                            portfolio_return += weight * stock_return
                            valid_weights_sum += weight
                    except Exception as e:
                        print(f"Error computing return for {symbol}: {e}")
                        continue
            
            # Adjust return by valid weight sum
            if valid_weights_sum > 0:
                # Renormalize return
                portfolio_return = portfolio_return / valid_weights_sum
            else:
                # If no valid weights, use zero return
                portfolio_return = 0.0
            
            # Compute new stock holdings value
            # Apply return only to non-cash portion
            if stock_holdings_value > 0:
                new_stock_value = stock_holdings_value * (1 + portfolio_return)
            else:
                new_stock_value = 0.0
            
            # Ensure cash is non-negative
            self.cash = max(0, self.cash)
            
            # Update portfolio value = stock value + cash
            new_portfolio_value = new_stock_value + self.cash
            
            # Avoid extreme values
            if np.isnan(new_portfolio_value) or np.isinf(new_portfolio_value):
                print(f"Warning: portfolio value invalid: {new_portfolio_value}, using old value")
                new_portfolio_value = old_portfolio_value
            
            # Ensure portfolio value does not become too small
            self.portfolio_value = max(new_portfolio_value, 0.01 * self.initial_capital)
            
            # Log portfolio value change
            if self.portfolio_value != old_portfolio_value:
                print(f"Portfolio value updated: {old_portfolio_value:.2f} -> {self.portfolio_value:.2f}")
                print(f"Holdings value: {new_stock_value:.2f}, cash: {self.cash:.2f}")
        
        # Update max portfolio value
        if self.portfolio_value > self.max_portfolio_value:
            self.max_portfolio_value = self.portfolio_value
        
        # Compute current drawdown
        if self.max_portfolio_value > 0:
            self.current_drawdown = (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value
            # Clip drawdown to reasonable range
            self.current_drawdown = np.clip(self.current_drawdown, 0.0, 0.99)
    
    def _calculate_reward(self):
        """Compute reward based on portfolio performance.

        Returns:
            float: Reward value
        """
        try:
            # Validate portfolio value
            if not hasattr(self, 'previous_portfolio_value') or self.previous_portfolio_value <= 0:
                print("Warning: previous portfolio value invalid, using 0 return")
                returns = 0.0
            else:
                # Compute return
                returns = (self.portfolio_value / self.previous_portfolio_value) - 1
                
                # Clip return to reasonable range
                returns = np.clip(returns, -0.5, 0.5)
            
            # ========== Vertical reward ==========
            # Return relative to initial capital
            initial_capital_return = (self.portfolio_value / self.initial_capital) - 1
            
            # Base vertical reward
            vertical_reward = 0.0
            
            # Read reward parameters
            capital_reward_pos_step = self.reward_params.get('capital_reward_pos_step', 2.0)
            capital_reward_neg_step = self.reward_params.get('capital_reward_neg_step', 0.5)
            
            # Vertical tiers based on initial capital
            # Positive: +10% adds capital_reward_pos_step
            if initial_capital_return > 0:
                # Compute positive reward
                positive_tier = int(initial_capital_return / 0.1)  # 10% per tier
                vertical_reward = positive_tier * capital_reward_pos_step
            else:
                # Compute negative reward
                negative_tier = int(abs(initial_capital_return) / 0.05)  # 5% per tier
                vertical_reward = -negative_tier * capital_reward_neg_step
            
            # ========== Horizontal reward ==========
            # Update consecutive up-day count
            if not hasattr(self, 'consecutive_up_days'):
                self.consecutive_up_days = 0
                
            if returns > 0:
                self.consecutive_up_days += 1
            else:
                self.consecutive_up_days = 0
                
            # Consecutive up-day reward (moderate scale)
            horizontal_reward = 0.0
            if self.consecutive_up_days >= 2:
                # Extra reward for 2+ consecutive up days
                horizontal_reward = min(self.consecutive_up_days * 0.5, 4.0)  # Cap at 4.0
            
            # ========== End-of-period reward ==========
            end_period_reward = 0.0
            
            # Check if last backtest day
            is_last_day = self.current_step == len(self.market_data) - 1
            
            # Extra reward if ending above initial capital
            if is_last_day and self.portfolio_value > self.initial_capital:
                # Set reward based on final return
                final_return_pct = (self.portfolio_value / self.initial_capital - 1) * 100
                end_period_reward = 2.0 if final_return_pct > 0 else 0.0
            
            # ========== Daily return reward ==========
            # Daily return as base reward
            base_reward_scale = self.reward_params.get('base_reward_scale', 3.0)
            base_reward = returns * base_reward_scale
            
            # Combine reward components
            reward = base_reward + vertical_reward + horizontal_reward + end_period_reward
            
            # Record reward components in history
            if hasattr(self, 'history') and isinstance(self.history, list) and len(self.history) > 0:
                current_record = self.history[-1]  # Current step record
                
                # Add reward info to current record
                current_record['returns'] = returns  # Raw returns
                current_record['vertical_reward'] = vertical_reward
                current_record['horizontal_reward'] = horizontal_reward  
                current_record['end_period_reward'] = end_period_reward
                current_record['base_reward'] = base_reward
                current_record['total_reward'] = reward
            
            # Validate reward value
            if np.isnan(reward) or np.isinf(reward):
                print(f"Warning: invalid reward: {reward}, set to 0")
                reward = 0.0
            
            # Clip reward to reasonable range
            reward_min = self.reward_params.get('reward_min', -7.0)
            reward_max = self.reward_params.get('reward_max', 14.0)
            reward = np.clip(reward, reward_min, reward_max)
            
            # Print diagnostics (every 20 steps)
            if self.current_step % 20 == 0 or is_last_day:
                print(f"Reward diagnostics - step {self.current_step}: return={returns:.4f}, reward={reward:.4f} "
                      f"(base={base_reward:.2f}, vertical={vertical_reward:.2f}, "
                      f"horizontal={horizontal_reward:.2f}, end={end_period_reward:.2f})")
                print(f"Portfolio value: {self.portfolio_value:.2f}, vs initial: {initial_capital_return:.2%}")
            
            return reward
            
        except Exception as e:
            print(f"Error computing reward: {e}")
            return 0.0
    
    def _calculate_portfolio_variance(self):
        """Compute portfolio variance from covariance matrix."""
        try:
            # Gather historical returns
            returns_data = []
            stock_symbols = []
            
            for symbol in self.current_stocks:
                if symbol in self.all_stock_data and symbol in self.current_weights:
                    df = self.all_stock_data[symbol]
                    # Use past 30 days for covariance
                    window_size = min(30, self.current_step)
                    start_idx = max(0, self.current_step - window_size)
                    end_idx = self.current_step + 1
                    
                    if start_idx < len(df) and end_idx <= len(df):
                        # Extract returns
                        returns_series = df.iloc[start_idx:end_idx]['daily_return'].values
                        
                        # Handle NaNs
                        if np.isnan(returns_series).any():
                            returns_series = np.nan_to_num(returns_series, nan=0.0)
                        
                        # Clip extreme values
                        returns_series = np.clip(returns_series, -0.2, 0.2)
                        
                        returns_data.append(returns_series)
                        stock_symbols.append(symbol)
            
            # If insufficient data, return 0
            if len(returns_data) < 2 or len(stock_symbols) < 2:
                return 0
            
            # Compute covariance matrix
            returns_matrix = np.array(returns_data)
            cov_matrix = np.cov(returns_matrix)
            
            # Validate covariance matrix
            if np.isnan(cov_matrix).any() or np.isinf(cov_matrix).any():
                print("Warning: covariance matrix invalid, returning 0")
                return 0
            
            # Extract weights for these stocks
            weights = np.array([self.current_weights[symbol] for symbol in stock_symbols])
            
            # Normalize weights to sum to 1
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)
            else:
                # If all weights are 0, use uniform distribution
                weights = np.ones(len(weights)) / len(weights)
            
            # Compute portfolio variance
            portfolio_variance = weights.T @ cov_matrix @ weights
            
            # Validate variance
            if np.isnan(portfolio_variance) or np.isinf(portfolio_variance) or portfolio_variance < 0:
                print(f"Warning: invalid portfolio variance: {portfolio_variance}, returning 0")
                return 0
            
            # Cap variance to avoid extreme values
            return min(portfolio_variance, 0.1)
            
        except Exception as e:
            print(f"Error computing portfolio variance: {e}")
            return 0
    
    def _calculate_drawdown(self):
        """Compute current drawdown from peak."""
        # Drawdown from peak
        drawdown = (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value
        
        return max(0, drawdown)
    
    def _check_stop_loss(self):
        """Dynamic stop-loss check (disabled, always False)."""
        # Stop-loss disabled
        self.stop_loss_triggered = False
        return False
    
    def _reset_after_stop_loss(self):
        """Reset after stop-loss (unused; stop-loss disabled)."""
        # Stop-loss disabled; no-op
        pass
    
    def _check_rebalance_trigger(self):
        """Check whether rebalance is triggered."""
        # Get current date
        current_date = pd.to_datetime(self.market_data.iloc[self.current_step]['date'])
        
        # Check random rebalance
        if self.config.get('environment', {}).get('random_rebalance', False):
            # Initialize next rebalance step if needed
            if not hasattr(self, 'next_rebalance_step') or self.next_rebalance_step is None:
                # Set mean and variance
                mean_days = self.config.get('environment', {}).get('random_rebalance_mean', 30)
                var_days = self.config.get('environment', {}).get('random_rebalance_var', 10)
                
                # Sample random interval (normal, min 5 days)
                random_days = max(5, int(np.random.normal(mean_days, var_days)))
                
                # Set next rebalance step
                self.next_rebalance_step = self.current_step + random_days
                print(f"Random rebalance initialized; next in {random_days} days")
                
                # Trigger rebalance on day 1
                if self.current_step == 0:
                    self.rebalance_count += 1
                    return True
            
            # Check whether rebalance time reached
            if self.current_step >= self.next_rebalance_step:
                # Sample next random interval
                mean_days = self.config.get('environment', {}).get('random_rebalance_mean', 30)
                var_days = self.config.get('environment', {}).get('random_rebalance_var', 10)
                random_days = max(5, int(np.random.normal(mean_days, var_days)))
                
                # Update next rebalance step
                self.next_rebalance_step = self.current_step + random_days
                print(f"Random rebalance triggered: {current_date}, next in {random_days} days")
                
                self.rebalance_count += 1
                return True
                
            return False
        
        # Check fixed-period rebalance (non-dynamic)
        if not self.config.get('environment', {}).get('dynamic_rebalance', True) and 'rebalance_period' in self.config.get('environment', {}):
            # Get rebalance period (days)
            rebalance_period = self.config['environment']['rebalance_period']
            
            # Initialize next rebalance step if needed
            if not hasattr(self, 'next_rebalance_step') or self.next_rebalance_step is None:
                self.next_rebalance_step = self.current_step + rebalance_period
                
                # Trigger rebalance on day 1
                if self.current_step == 0:
                    self.rebalance_count += 1
                    return True
            
            # Check if rebalance period reached
            if self.current_step >= self.next_rebalance_step:
                # Update next rebalance step
                self.next_rebalance_step = self.current_step + rebalance_period
                print(f"Fixed-period rebalance triggered: {current_date}, period {rebalance_period} days")
                
                self.rebalance_count += 1
                return True
                
            return False
        
        # Default: rebalance on first trading day of month
        if current_date.day <= 5:  # Extend window for holidays
            # Check first trading day of the month
            month_start = pd.Timestamp(year=current_date.year, month=current_date.month, day=1)
            # Get all trading days so far this month
            month_trading_days = [pd.to_datetime(self.market_data.iloc[i]['date']) 
                                 for i in range(self.current_step+1) 
                                 if pd.to_datetime(self.market_data.iloc[i]['date']).month == current_date.month
                                 and pd.to_datetime(self.market_data.iloc[i]['date']).year == current_date.year]
            
            # If current date is the first trading day
            if month_trading_days and month_trading_days[0] == current_date:
                print(f"Month-start rebalance triggered: {current_date}")
                self.rebalance_count += 1
                return True
                
        return False
    
    def _get_market_features(self, current_date):
        """Get market-level features."""
        # Safety check: ensure current_step within range
        if self.current_step >= len(self.market_data):
            # If out of range, use last valid index
            safe_step = len(self.market_data) - 1
            market_row = self.market_data.iloc[safe_step]
        else:
            # Get current row from market data
            market_row = self.market_data.iloc[self.current_step]
        
        # Extract market features
        market_features = []
        
        # Basic market features
        market_features.append(market_row.get('daily_return', 0))
        market_features.append(market_row.get('volatility', 0))
        market_features.append(market_row.get('momentum', 0))
        
        # Add market sentiment if available
        if 'sentiment' in market_row:
            market_features.append(market_row['sentiment'])
        
        # Add market trend if available
        if 'trend' in market_row:
            market_features.append(market_row['trend'])
            
        return np.array(market_features, dtype=np.float32)
        
    def _get_stock_features(self, current_date):
        """Get per-stock features for the current portfolio."""
        all_stock_features = []
        
        for symbol in self.current_stocks:
            if symbol not in self.current_stock_data:
                # If no data, use zero vector
                # Feature dimension assumed standard; adjust via config if needed
                stock_features = np.zeros(10)  # Default 10 features
            else:
                stock_data = self.current_stock_data[symbol]
                
                # Get nearest row for current date
                stock_df = stock_data[stock_data['date'] <= current_date]
                
                if len(stock_df) == 0:
                    # If no row matches, use zero vector
                    stock_features = np.zeros(10)
                else:
                    # Get nearest-date data
                    stock_row = stock_df.iloc[-1]
                    
                    # Extract basic stock features
                    stock_features = []
                    
                    # Daily return
                    stock_features.append(stock_row.get('daily_return', 0))
                    
                    # Volatility
                    stock_features.append(stock_row.get('volatility', 0))
                    
                    # Valuation metrics
                    if 'pe_ratio' in stock_row:
                        stock_features.append(stock_row['pe_ratio'])
                        
                    # Technical indicators
                    if 'rsi_14' in stock_row:
                        stock_features.append(stock_row['rsi_14'])
                    
                    if 'macd' in stock_row:
                        stock_features.append(stock_row['macd'])
                    
                    if 'bb_width' in stock_row:
                        stock_features.append(stock_row['bb_width'])
                    
                    if 'atr_14' in stock_row:
                        stock_features.append(stock_row['atr_14'])
                    
                    # Relative strength vs benchmark
                    if 'relative_strength' in stock_row:
                        stock_features.append(stock_row['relative_strength'])
                    
                    # Volume-related indicators
                    if 'volume_change' in stock_row:
                        stock_features.append(stock_row['volume_change'])
            
            # Append features to overall vector
            all_stock_features.extend(stock_features)
            
        return np.array(all_stock_features, dtype=np.float32)
    
    def _get_portfolio_features(self):
        """Get portfolio-level features."""
        portfolio_features = []
        
        # Add current portfolio value
        portfolio_features.append(self.portfolio_value / self.initial_capital)
        
        # Add current max drawdown
        portfolio_features.append(self.current_drawdown)
        
        # Add step-to-step change rate
        if hasattr(self, 'previous_portfolio_value') and self.previous_portfolio_value > 0:
            portfolio_change = (self.portfolio_value / self.previous_portfolio_value) - 1
        else:
            portfolio_change = 0
        portfolio_features.append(portfolio_change)
        
        # Add cash ratio (<= 100%)
        cash_ratio = min(1.0, self.cash / self.portfolio_value) if self.portfolio_value > 0 else 0
        portfolio_features.append(cash_ratio)
        
        # Add stop-loss and rebalance status
        portfolio_features.append(1.0 if self.stop_loss_triggered else 0.0)
        
        return np.array(portfolio_features, dtype=np.float32)
        
    def _record_state(self, info):
        """Record current state into history."""
        # Get current date safely from market_data
        try:
            # If info already has day, use it
            if 'day' in info and info['day'] is not None:
                current_date = info['day']
            else:
                # Ensure current_step is valid index
                if 0 <= self.current_step < len(self.market_data):
                    if isinstance(self.market_data, pd.DataFrame):
                        current_date = self.market_data.iloc[self.current_step]['date'] if 'date' in self.market_data.columns else None
                    else:
                        # If market_data is not a DataFrame, handle differently
                        current_date = f"Step_{self.current_step}"
                else:
                    # If index out of range, use fallback
                    current_date = f"Step_{self.current_step}"
        except Exception as e:
            print(f"Error getting current date: {e}")
            current_date = f"Step_{self.current_step}"
        
        # Compute cash ratio (<= 100%)
        cash_ratio = min(1.0, self.cash / self.portfolio_value) if self.portfolio_value > 0 else 0
        
        # Record detailed state info
        state_record = {
            'date': current_date,
            'portfolio_value': self.portfolio_value,
            'initial_capital': self.initial_capital,
            'return_pct': (self.portfolio_value / self.initial_capital - 1) * 100,  # Cumulative return (%)
            'daily_return': (self.portfolio_value / self.previous_portfolio_value - 1) if hasattr(self, 'previous_portfolio_value') and self.previous_portfolio_value > 0 else 0,  # Daily return
            'weights': self.current_weights.copy(),
            'stocks': self.current_stocks.copy(),
            'cash': self.cash,
            'cash_ratio': cash_ratio,  # Cash ratio (0-100%)
            'drawdown': self.current_drawdown * 100,  # Drawdown (%)
            'max_value': self.max_portfolio_value,  # Max portfolio value
            'transaction_cost': info.get('transaction_cost', 0),
            'stop_loss': info.get('stop_loss_triggered', False),
            'rebalance': info.get('rebalance_triggered', False),
            'holdings': self.holdings.copy() if hasattr(self, 'holdings') and self.holdings else {}
        }
        
        # Add ATR-related info
        if hasattr(self, 'atr_history') and len(self.atr_history) > 0 and self.current_step < len(self.atr_history):
            state_record['atr'] = self.atr_history[self.current_step]
            state_record['stop_loss_threshold'] = self.stop_loss_threshold_history[self.current_step]
        else:
            state_record['atr'] = None
            state_record['stop_loss_threshold'] = self.original_stop_loss_threshold if hasattr(self, 'original_stop_loss_threshold') else None
        
        # Append to history
        if not hasattr(self, 'history') or not isinstance(self.history, list):
            self.history = []
        self.history.append(state_record)
        
        # Update previous portfolio value for next step
        self.previous_portfolio_value = self.portfolio_value
    
    def export_history(self, file_path=None):
        """Export trade history to a CSV file."""
        if not self.history:
            print("No history to export")
            return None
        
        # Prepare base data
        basic_data = []
        for record in self.history:
            row = {
                'date': record['date'],
                'portfolio_value': record['portfolio_value'],
                'return_pct': record.get('return_pct', (record['portfolio_value'] / self.initial_capital - 1) * 100),
                'daily_return': record.get('daily_return', 0),
                'cash': record.get('cash', 0),
                'cash_ratio': record.get('cash_ratio', 0),
                'drawdown': record.get('drawdown', 0),
                'transaction_cost': record['transaction_cost'],
                'stop_loss': record['stop_loss'],
                'rebalance': record['rebalance'],
                'stock_count': len(record['stocks'])
            }
            basic_data.append(row)
        
        # Build base DataFrame
        df_basic = pd.DataFrame(basic_data)
        
        # If no file path, build default path
        if file_path is None:
            # Create results directory
            results_dir = 'results'
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(results_dir, f"portfolio_history_{timestamp}.csv")
        else:
            # Ensure target directory exists
            output_dir = os.path.dirname(file_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
        
        # Save base data
        df_basic.to_csv(file_path, index=False)
        print(f"Base portfolio history saved to: {file_path}")
        
        # Build weights data
        # Get all tickers encountered
        all_symbols = set()
        for record in self.history:
            all_symbols.update(record['weights'].keys())
        
        # Add weights and holdings per ticker
        weights_data = []
        for record in self.history:
            row = {'date': record['date']}
            # Add weights
            for symbol in all_symbols:
                row[f'weight_{symbol}'] = record['weights'].get(symbol, 0)
            # Add holdings
            if 'holdings' in record and record['holdings']:
                for symbol in all_symbols:
                    row[f'holding_{symbol}'] = record['holdings'].get(symbol, 0)
            weights_data.append(row)
        
        # Build weights DataFrame
        df_weights = pd.DataFrame(weights_data)
        
        # Save weights data
        weights_file_path = file_path.replace('.csv', '_weights.csv')
        df_weights.to_csv(weights_file_path, index=False)
        print(f"Weights and holdings history saved to: {weights_file_path}")
        
        return file_path

    def _calculate_portfolio_weights(self, state):
        """Calculate portfolio weights."""
        if self.config['portfolio'].get('use_mean_var', False):
            # Use mean-variance optimization
            return self._mean_variance_optimization()
        elif self.config['portfolio'].get('use_momentum', False):
            # Use momentum strategy
            return self._momentum_strategy()
        elif self.config['portfolio'].get('equal_weight', False):
            # Use equal-weight strategy
            return self._equal_weight_strategy()
        else:
            # Use RL strategy
            return self.agent.predict(state)
            
    def _mean_variance_optimization(self):
        """Mean-variance optimization strategy."""
        window = 60
        returns = pd.DataFrame()
        
        # Compute historical returns per stock
        for symbol in self.current_stocks:
            prices = self.data[symbol]['close'][-window:]
            returns[symbol] = prices.pct_change().dropna()
        
        # Compute expected returns and covariance
        mu = returns.mean()
        sigma = returns.cov()
        
        try:
            # Use scipy for quadratic optimization
            from scipy.optimize import minimize
            
            def objective(weights):
                portfolio_return = np.sum(weights * mu)
                portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(sigma, weights)))
                # Maximize Sharpe ratio
                return -(portfolio_return - 0.02) / portfolio_risk  # Assume 2% risk-free rate
            
            n_assets = len(self.current_stocks)
            constraints = (
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
            )
            bounds = tuple((0, 1) for _ in range(n_assets))  # Weights in [0,1]
            
            result = minimize(
                objective,
                x0=np.array([1/n_assets] * n_assets),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if result.success:
                weights = result.x
                return dict(zip(self.current_stocks, weights))
            else:
                print("Mean-variance optimization failed; using equal weights")
                return self._equal_weight_strategy()
                
        except Exception as e:
            print(f"Mean-variance optimization error: {e}")
            return self._equal_weight_strategy()
    
    def _momentum_strategy(self):
        """Momentum strategy."""
        momentum_scores = {}
        window = 20
        
        for symbol in self.current_stocks:
            try:
                prices = self.data[symbol]['close'][-window:]
                # Momentum score (current / 20-day-ago - 1)
                momentum = (prices.iloc[-1] / prices.iloc[0] - 1)
                momentum_scores[symbol] = momentum
            except Exception as e:
                print(f"Error computing momentum for {symbol}: {e}")
                momentum_scores[symbol] = 0
        
        # Normalize momentum scores
        total_score = sum(max(0, score) for score in momentum_scores.values())
        if total_score > 0:
            weights = {symbol: max(0, score) / total_score 
                      for symbol, score in momentum_scores.items()}
        else:
            # If all momentum scores are negative, use equal weights
            weights = self._equal_weight_strategy()
        
        return weights
    
    def _equal_weight_strategy(self):
        """Equal-weight strategy."""
        equal_weight = 1.0 / len(self.current_stocks) if self.current_stocks else 0
        return {symbol: equal_weight for symbol in self.current_stocks} 

    def _ensure_valid_state(self):
        """Ensure environment state is valid."""
        # Ensure cash is non-negative
        if self.cash < 0:
            print(f"Warning: negative cash ({self.cash:.2f}), resetting to 0")
            self.cash = 0
            
            # Recompute portfolio value
            holdings_value = sum(self.holdings.values()) if self.holdings else 0
            self.portfolio_value = holdings_value + self.cash
            
            # Recheck weights after cash change
            total_value = self.portfolio_value
            if total_value > 0:
                self.current_weights = {symbol: value/total_value for symbol, value in self.holdings.items()}
            else:
                self.current_weights = {}
        
        # Ensure portfolio value is non-negative
        if self.portfolio_value < 0:
            print(f"Warning: negative portfolio value ({self.portfolio_value:.2f}), reset to cash value")
            self.portfolio_value = self.cash

