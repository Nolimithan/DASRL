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
        self.stop_loss_threshold = config['environment']['stop_loss_threshold']
        
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
        self.holdings = {}
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
        self.holdings = {}
        self.current_stocks = []
        self.stop_loss_triggered = False
        
        # Reset history
        self.history = []
        
        # Reset statistics
        self.rebalance_count = 0
        self.stop_loss_count = 0
        self.max_drawdown = 0
        self.current_drawdown = 0
        self.transaction_cost_total = 0
        
        # Select initial portfolio stocks
        self._select_portfolio_stocks()
        
        # Get initial state
        initial_state = self._get_state()
        
        return initial_state
    
    def step(self, action):
        """Execute a trading action."""
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
        
        # Update portfolio value
        self._update_portfolio_value()
        
        # Check stop-loss
        stop_loss_triggered = self._check_stop_loss()
        
        # Check if rebalance is needed (select new stocks)
        rebalance_triggered = self._check_rebalance_trigger()
        
        # Reselect stocks if stop-loss or rebalance triggered
        if stop_loss_triggered or rebalance_triggered:
            self._select_portfolio_stocks()
        
        # Compute reward for this step
        reward = self._calculate_reward()
        
        # Update portfolio date
        self.current_step += 1
        
        # Determine episode termination
        done = self.current_step >= len(self.market_data) - 1
        
        # Get next state
        next_state = self._get_state()
        
        # Return info dict
        info = {
            'portfolio_value': self.portfolio_value,
            'weights': self.current_weights,
            'transaction_cost': transaction_cost,
            'stop_loss_triggered': stop_loss_triggered,
            'rebalance_triggered': rebalance_triggered,
            'current_stocks': self.current_stocks,
            'day': self.market_data.iloc[self.current_step]['date'] if not done else None
        }
        
        # Record state
        self._record_state(info)
        
        return next_state, reward, done, info
    
    def _get_state(self):
        """Build state vector from market, stock, and portfolio features."""
        # Get current date
        current_date = self.market_data.iloc[self.current_step]['date']
        
        # 1. Global market features
        market_features = self._get_market_features(current_date)
        
        # 2. Local stock features
        stock_features = self._get_stock_features(current_date)
        
        # 3. Portfolio features
        portfolio_features = self._get_portfolio_features()
        
        # Combine all features
        state = np.concatenate([market_features, stock_features, portfolio_features])
        
        # Ensure state dimension matches observation space
        actual_dim = len(state)
        expected_dim = self.observation_space.shape[0]
        
        if actual_dim != expected_dim:
            print(f"Warning: state dim mismatch, current: {actual_dim}, expected: {expected_dim}")
            # Truncate if too large; pad if too small
            if actual_dim > expected_dim:
                state = state[:expected_dim]
            else:
                padding = np.zeros(expected_dim - actual_dim, dtype=np.float32)
                state = np.concatenate([state, padding])
                
        return state
    
    def _select_portfolio_stocks(self):
        """Select portfolio stocks based on LIM."""
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
            try:
                # Try LIM to select top stocks
                top_stocks = self.lim_calculator.get_top_stocks(
                    self.data_loader.all_stock_data,  # Use all_stock_data
                    self.data_loader.benchmark['daily_return'],  # Benchmark returns
                    current_date,  # Current date
                    top_n=self.portfolio_size
                )
                
                # Randomly fill if LIM selection is insufficient
                if len(top_stocks) < self.portfolio_size:
                    print(f"LIM selected {len(top_stocks)} stocks, need: {self.portfolio_size}")
                    
                    # Find unselected stocks
                    remaining_stocks = [s for s in avaSIMble_stocks if s not in top_stocks]
                    
                    # Add needed stocks
                    additional_count = min(self.portfolio_size - len(top_stocks), len(remaining_stocks))
                    if additional_count > 0:
                        print(f"Randomly added {additional_count} stocks")
                        additional_stocks = np.random.choice(
                            remaining_stocks, 
                            size=additional_count,
                            replace=False
                        ).tolist()
                        top_stocks.extend(additional_stocks)
                
                self.current_stocks = top_stocks
            except Exception as e:
                print(f"LIM selection failed: {str(e)}")
                print("Falling back to random selection")
                # Random selection fallback
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
        # Get current date
        current_date = self.market_data.iloc[self.current_step]['date']
        
        # Initialize holdings if needed
        if not hasattr(self, 'holdings') or self.holdings is None:
            self.holdings = {}
            
        # Compute current net asset value
        total_value = self.portfolio_value
        
        # Compute target holding values
        target_values = {}
        for symbol, weight in target_weights.items():
            target_values[symbol] = total_value * weight
            
        # Compute required adjustments
        adjustments = {}
        for symbol in set(list(self.holdings.keys()) + list(target_weights.keys())):
            current_value = self.holdings.get(symbol, 0)
            target_value = target_values.get(symbol, 0)
            adjustments[symbol] = target_value - current_value
            
        # Compute total buy and sell amounts
        total_buy = sum([adj for adj in adjustments.values() if adj > 0])
        total_sell = sum([-adj for adj in adjustments.values() if adj < 0])
        
        # Compute transaction cost
        buy_cost = total_buy * self.commission_rate
        sell_cost = total_sell * self.commission_rate
        transaction_cost = buy_cost + sell_cost
        
        # Update holdings
        for symbol, adjustment in adjustments.items():
            if symbol in self.holdings:
                self.holdings[symbol] += adjustment
                if self.holdings[symbol] < 0.01:  # Avoid tiny values
                    del self.holdings[symbol]
            else:
                if adjustment > 0:
                    self.holdings[symbol] = adjustment
        
        # Update cash
        self.cash = self.cash - total_buy + total_sell - transaction_cost
        
        # Accumulate total transaction cost
        self.transaction_cost_total += transaction_cost
        
        # Update today's trade record
        self.today_trades = adjustments
        
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
            portfolio_return = 0.0
            valid_weights_sum = 0.0  # Track valid weight sum
            
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
            
            # Update portfolio value; avoid negative/NaN
            new_portfolio_value = old_portfolio_value * (1 + portfolio_return)
            
            # Avoid extreme values
            if np.isnan(new_portfolio_value) or np.isinf(new_portfolio_value):
                print(f"Warning: portfolio value invalid: {new_portfolio_value}, using old value")
                new_portfolio_value = old_portfolio_value
            
            # Ensure portfolio value does not become too small
            self.portfolio_value = max(new_portfolio_value, 0.01 * self.initial_capital)
        
        # Update max portfolio value
        if self.portfolio_value > self.max_portfolio_value:
            self.max_portfolio_value = self.portfolio_value
        
        # Compute current drawdown
        if self.max_portfolio_value > 0:
            self.current_drawdown = (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value
            # Clip drawdown to reasonable range
            self.current_drawdown = np.clip(self.current_drawdown, 0.0, 0.99)
    
    def _calculate_reward(self):
        """Compute reward based on returns, variance, turnover, and drawdown.

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
            
            # Compute variance penalty
            variance_penalty = self._calculate_portfolio_variance()
            
            # Turnover penalty: sum of absolute weight changes / 2
            turnover_penalty = 0.0
            if hasattr(self, 'previous_weights') and self.previous_weights:
                all_symbols = set(list(self.current_weights.keys()) + list(self.previous_weights.keys()))
                for symbol in all_symbols:
                    prev_weight = self.previous_weights.get(symbol, 0.0)
                    current_weight = self.current_weights.get(symbol, 0.0)
                    turnover_penalty += abs(current_weight - prev_weight)
                turnover_penalty = turnover_penalty / 2.0
            
            # Drawdown penalty
            drawdown_penalty = self._calculate_drawdown()
            
            # Validate penalty terms
            if np.isnan(variance_penalty) or np.isinf(variance_penalty):
                print(f"Warning: invalid variance penalty: {variance_penalty}, set to 0")
                variance_penalty = 0.0
            
            if np.isnan(turnover_penalty) or np.isinf(turnover_penalty):
                print(f"Warning: invalid turnover penalty: {turnover_penalty}, set to 0")
                turnover_penalty = 0.0
            
            if np.isnan(drawdown_penalty) or np.isinf(drawdown_penalty):
                print(f"Warning: invalid drawdown penalty: {drawdown_penalty}, set to 0")
                drawdown_penalty = 0.0
            
            # Read reward parameters
            lambda_1 = self.reward_params.get('lambda_1', 0.5)
            lambda_2 = self.reward_params.get('lambda_2', 0.2)
            lambda_3 = self.reward_params.get('lambda_3', 1.0)
            
            # Ensure coefficients are non-negative
            lambda_1 = max(0, lambda_1)
            lambda_2 = max(0, lambda_2)
            lambda_3 = max(0, lambda_3)
            
            # Combined reward
            reward = returns - \
                    lambda_1 * variance_penalty - \
                    lambda_2 * turnover_penalty - \
                    lambda_3 * drawdown_penalty
            
            # Validate reward value
            if np.isnan(reward) or np.isinf(reward):
                print(f"Warning: invalid reward: {reward}, set to 0")
                reward = 0.0
            
            # Clip reward to reasonable range
            reward = np.clip(reward, -10.0, 10.0)
            
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
        """Check whether stop-loss is triggered."""
        # Compute current drawdown
        self.current_drawdown = (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value
        self.current_drawdown = np.clip(self.current_drawdown, 0.0, 0.99)  # Keep drawdown in range
        
        # Use dynamic stop-loss threshold; increase with consecutive triggers
        effective_threshold = min(self.stop_loss_threshold * (1 + 0.2 * self.stop_loss_count), 0.5)
        
        # Trigger stop-loss if drawdown exceeds threshold
        if self.current_drawdown > effective_threshold:
            print(f"Stop-loss triggered: drawdown {self.current_drawdown:.2%} > threshold {effective_threshold:.2%}")
            self.stop_loss_triggered = True
            self.stop_loss_count += 1
            
            # Reset portfolio state partially
            self._reset_after_stop_loss()
            
            return True
        
        self.stop_loss_triggered = False
        return False
    
    def _reset_after_stop_loss(self):
        """Reset portfolio state after stop-loss."""
        # Lower max portfolio value to allow recovery
        self.max_portfolio_value = max(self.portfolio_value * 1.2, self.initial_capital * 0.5)
        
        # Reset current drawdown
        self.current_drawdown = (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value
        
        # Increase cash ratio to reduce risk
        previous_cash = self.cash
        self.cash = max(self.cash, self.portfolio_value * 0.3)  # Keep at least 30% cash
        
        print(f"Post stop-loss reset: max value {self.max_portfolio_value:.2f}, drawdown {self.current_drawdown:.2%}")
        print(f"Cash increased: {previous_cash:.2f} -> {self.cash:.2f} ({self.cash/self.portfolio_value:.2%})")
        
        # Every 3 triggers, reset stop-loss count
        if self.stop_loss_count % 3 == 0:
            print(f"Stop-loss triggered {self.stop_loss_count} times, resetting count")
            self.stop_loss_count = 0
    
    def _check_rebalance_trigger(self):
        """Check whether rebalance is triggered."""
        # Get current date
        current_date = pd.to_datetime(self.market_data.iloc[self.current_step]['date'])
        
        # Periodic rebalance (e.g., first trading day of month)
        if current_date.day == 1:
            print(f"Periodic rebalance triggered: {current_date}")
            self.rebalance_count += 1
            return True
            
        # Trigger rebalance if holdings not in top list
        if hasattr(self, 'lim_calculator') and self.lim_calculator is not None:
            try:
                # Get current top stocks
                current_date_str = self.market_data.iloc[self.current_step]['date']
                top_stocks = self.lim_calculator.get_top_stocks(
                    self.data_loader.all_stock_data,
                    self.data_loader.benchmark['daily_return'],
                    current_date_str,
                    top_n=self.portfolio_size * 2  # Expand candidate pool
                )
                
                # If no stocks selected, skip rebalance check
                if not top_stocks:
                    print("LIM selected no stocks; skipping rebalance check")
                    return False
                
                # Check if current holdings are in top list
                stocks_to_replace = [s for s in self.current_stocks if s not in top_stocks]
                
                # Trigger rebalance if >50% need replacement
                if len(stocks_to_replace) > len(self.current_stocks) / 2:
                    print(f"LIM-based rebalance: replace {len(stocks_to_replace)} stocks")
                    self.rebalance_count += 1
                    return True
            except Exception as e:
                print(f"Error during rebalance check: {e}")
                return False
                
        return False
    
    def _get_market_features(self, current_date):
        """Get market-level features."""
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
        
        # Add cash ratio
        cash_ratio = self.cash / self.portfolio_value if self.portfolio_value > 0 else 0
        portfolio_features.append(cash_ratio)
        
        # Add stop-loss and rebalance status
        portfolio_features.append(1.0 if self.stop_loss_triggered else 0.0)
        
        return np.array(portfolio_features, dtype=np.float32)
        
    def _record_state(self, info):
        """Record current state into history."""
        # Get current date
        current_date = self.market_data.iloc[self.current_step]['date']
        
        # Record state
        state_record = {
            'date': current_date,
            'portfolio_value': self.portfolio_value,
            'weights': self.current_weights.copy(),
            'stocks': self.current_stocks.copy(),
            'transaction_cost': info.get('transaction_cost', 0),
            'stop_loss': info.get('stop_loss_triggered', False),
            'rebalance': info.get('rebalance_triggered', False)
        }
        
        # Append to history
        self.history.append(state_record)
        
        # Update previous portfolio value for next step
        self.previous_portfolio_value = self.portfolio_value 

