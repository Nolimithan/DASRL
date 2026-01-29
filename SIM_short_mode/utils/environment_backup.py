"""
Trading environment module based on OpenAI Gym.
"""

import numpy as np
import pandas as pd
import gym
from gym import spaces
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

class TradingEnvironment(gym.Env):
    """Trading environment for reinforcement learning."""
    
    def __init__(self, config, data_loader, lim_calculator):
        """Initialize the trading environment."""
        super(TradingEnvironment, self).__init__()
        
        self.config = config
        self.data_loader = data_loader
        self.lim_calculator = lim_calculator
        
        self.portfolio_size = config['environment']['portfolio_size']
        self.initial_capital = config['environment']['initial_capital']
        self.commission_rate = config['environment']['commission_rate']
        self.stop_loss_threshold = config['environment']['stop_loss_threshold']
        
        self.reward_params = config['rl']['reward_params'] if 'rl' in config and 'reward_params' in config['rl'] else {
            'lambda_1': 0.5,
            'lambda_2': 0.5,
            'lambda_3': 1.0
        }
        
        self.action_space = gym.spaces.Box(
            low=0, high=1,
            shape=(self.portfolio_size,),
            dtype=np.float32
        )
        
        self.state_dim = self._calculate_state_dim()
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32
        )
        
        self.portfolio_value = self.initial_capital
        self.cash = self.initial_capital
        self.max_portfolio_value = self.initial_capital
        self.current_weights = {}
        self.previous_weights = {}
        self.holdings = {}
        self.current_stocks = []
        self.previous_portfolio_value = self.initial_capital
        
        self.rebalance_count = 0
        self.stop_loss_count = 0
        self.current_drawdown = 0
        self.transaction_cost_total = 0
        
        self.history = []
        
        self.stop_loss_triggered = False
        self.today_trades = {}
        
    def _calculate_state_dim(self):
        """Calculate state vector dimension."""
        market_feature_count = 3
        stock_feature_count = 2
        portfolio_feature_count = 5
        
        total_dim = market_feature_count + (stock_feature_count * self.portfolio_size) + portfolio_feature_count
        
        print(f"Calculated state dimension: {total_dim}")
        return total_dim
    
    def reset(self, start_date=None, end_date=None):
        """Reset environment state and initialize portfolio."""
        if start_date is None:
            start_date = self.config['data']['start_date']
        if end_date is None:
            end_date = self.config['data']['end_date']
            
        self.start_date_str = start_date
        self.end_date_str = end_date
            
        print(f"Loading data from {start_date} to {end_date}...")
        
        self.data_loader.set_date_range(start_date, end_date)
        
        market_data = self.data_loader.benchmark
        
        if market_data is None or len(market_data) == 0:
            raise ValueError("Failed to load market data. Check date range and data source.")
            
        self.market_data = market_data
        
        self.all_stock_data = self.data_loader.all_stock_data
        
        if not self.all_stock_data:
            raise ValueError("No stock data loaded. Check symbols and data source.")
            
        self.current_step = 0
        
        self.portfolio_value = self.initial_capital
        self.cash = self.initial_capital
        self.max_portfolio_value = self.initial_capital
        self.current_weights = {}
        self.previous_weights = {}
        self.holdings = {}
        self.current_stocks = []
        self.stop_loss_triggered = False
        
        self.history = []
        
        self.rebalance_count = 0
        self.stop_loss_count = 0
        self.max_drawdown = 0
        self.current_drawdown = 0
        self.transaction_cost_total = 0
        
        self._select_portfolio_stocks()
        
        initial_state = self._get_state()
        
        return initial_state
    
    def step(self, action):
        """Execute a trading action."""
        self.previous_weights = self.current_weights.copy() if self.current_weights else {}
        
        if len(action) != len(self.current_stocks):
            raise ValueError(f"Action dimension ({len(action)}) does not match portfolio size ({len(self.current_stocks)})")
            
        if np.sum(action) > 0:
            action = action / np.sum(action)
        else:
            action = np.ones(len(action)) / len(action)
            
        target_weights = dict(zip(self.current_stocks, action))
        
        target_weights = self._apply_trading_constraints(target_weights)
        
        transaction_cost = self._execute_trades(target_weights)
        
        self.current_weights = target_weights
        
        self._update_portfolio_value()
        
        stop_loss_triggered = self._check_stop_loss()
        
        rebalance_triggered = self._check_rebalance_trigger()
        
        if stop_loss_triggered or rebalance_triggered:
            self._select_portfolio_stocks()
        
        reward = self._calculate_reward()
        
        self.current_step += 1
        
        done = self.current_step >= len(self.market_data) - 1
        
        next_state = self._get_state()
        
        info = {
            'portfolio_value': self.portfolio_value,
            'weights': self.current_weights,
            'transaction_cost': transaction_cost,
            'stop_loss_triggered': stop_loss_triggered,
            'rebalance_triggered': rebalance_triggered,
            'current_stocks': self.current_stocks,
            'day': self.market_data.iloc[self.current_step]['date'] if not done else None
        }
        
        self._record_state(info)
        
        return next_state, reward, done, info
    
    def _get_state(self):
        """Build state vector from market and portfolio features."""
        current_date = self.market_data.iloc[self.current_step]['date']
        
        market_features = self._get_market_features(current_date)
        
        stock_features = self._get_stock_features(current_date)
        
        portfolio_features = self._get_portfolio_features()
        
        state = np.concatenate([market_features, stock_features, portfolio_features])
        
        actual_dim = len(state)
        expected_dim = self.observation_space.shape[0]
        
        if actual_dim != expected_dim:
            print(f"Warning: state dimension mismatch: {actual_dim} vs {expected_dim}")
            if actual_dim > expected_dim:
                state = state[:expected_dim]
            else:
                padding = np.zeros(expected_dim - actual_dim, dtype=np.float32)
                state = np.concatenate([state, padding])
                
        return state
    
    def _select_portfolio_stocks(self):
        """Select portfolio stocks based on LIM scores."""
        current_date = self.market_data.iloc[self.current_step]['date']
        
        avaSIMble_stocks = list(self.data_loader.all_stock_data.keys())
        
        avaSIMble_stocks = [s for s in avaSIMble_stocks if not (s.startswith('159') or s.startswith('512'))]
        
        if len(avaSIMble_stocks) < self.portfolio_size:
            print(f"Warning: available stocks ({len(avaSIMble_stocks)}) less than portfolio size ({self.portfolio_size})")
            self.current_stocks = avaSIMble_stocks
            print(f"Using all {len(self.current_stocks)} available stocks")
        else:
            try:
                top_stocks = self.lim_calculator.get_top_stocks(
                    self.data_loader.all_stock_data,
                    self.data_loader.benchmark['daily_return'],
                    current_date,
                    top_n=self.portfolio_size
                )
                
                if len(top_stocks) < self.portfolio_size:
                    print(f"LIM selected {len(top_stocks)} stocks; need {self.portfolio_size}")
                    
                    remaining_stocks = [s for s in avaSIMble_stocks if s not in top_stocks]
                    
                    additional_count = min(self.portfolio_size - len(top_stocks), len(remaining_stocks))
                    if additional_count > 0:
                        print(f"Randomly adding {additional_count} stocks")
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
                self.current_stocks = np.random.choice(
                    avaSIMble_stocks, 
                    size=min(self.portfolio_size, len(avaSIMble_stocks)),
                    replace=False
                ).tolist()
        
        print(f"Selected {len(self.current_stocks)} stocks for the portfolio")
        
        self.current_stock_data = {}
        for symbol in self.current_stocks:
            if symbol in self.data_loader.all_stock_data:
                self.current_stock_data[symbol] = self.data_loader.all_stock_data[symbol]
        
        equal_weight = 1.0 / len(self.current_stocks) if self.current_stocks else 0
        self.current_weights = {symbol: equal_weight for symbol in self.current_stocks}
    
    def _apply_trading_constraints(self, target_weights):
        """Apply trading constraints to target weights."""
        if not self.previous_weights:
            return target_weights
            
        constrained_weights = {}
        
        total_weight_change = 0
        for symbol in set(list(self.previous_weights.keys()) + list(target_weights.keys())):
            prev_weight = self.previous_weights.get(symbol, 0)
            target_weight = target_weights.get(symbol, 0)
            total_weight_change += abs(target_weight - prev_weight)
        
        max_allowed_change = 0.2
        
        if total_weight_change > max_allowed_change:
            reduction_factor = max_allowed_change / total_weight_change
            
            for symbol in set(list(self.previous_weights.keys()) + list(target_weights.keys())):
                prev_weight = self.previous_weights.get(symbol, 0)
                target_weight = target_weights.get(symbol, 0)
                
                weight_change = target_weight - prev_weight
                constrained_weight = prev_weight + weight_change * reduction_factor
                
                constrained_weight = max(0, constrained_weight)
                
                if constrained_weight > 0:
                    constrained_weights[symbol] = constrained_weight
        else:
            constrained_weights = target_weights.copy()
            
        total_weight = sum(constrained_weights.values())
        if total_weight > 0:
            for symbol in constrained_weights:
                constrained_weights[symbol] /= total_weight
                
        return constrained_weights
    
    def _execute_trades(self, target_weights):
        """Execute trades to reach target weights."""
        current_date = self.market_data.iloc[self.current_step]['date']
        
        if not hasattr(self, 'holdings') or self.holdings is None:
            self.holdings = {}
            
        total_value = self.portfolio_value
        
        target_values = {}
        for symbol, weight in target_weights.items():
            target_values[symbol] = total_value * weight
            
        adjustments = {}
        for symbol in set(list(self.holdings.keys()) + list(target_weights.keys())):
            current_value = self.holdings.get(symbol, 0)
            target_value = target_values.get(symbol, 0)
            adjustments[symbol] = target_value - current_value
            
        total_buy = sum([adj for adj in adjustments.values() if adj > 0])
        total_sell = sum([-adj for adj in adjustments.values() if adj < 0])
        
        buy_cost = total_buy * self.commission_rate
        sell_cost = total_sell * self.commission_rate
        transaction_cost = buy_cost + sell_cost
        
        for symbol, adjustment in adjustments.items():
            if symbol in self.holdings:
                self.holdings[symbol] += adjustment
                if self.holdings[symbol] < 0.01:
                    del self.holdings[symbol]
            else:
                if adjustment > 0:
                    self.holdings[symbol] = adjustment
        
        self.cash = self.cash - total_buy + total_sell - transaction_cost
        
        self.transaction_cost_total += transaction_cost
        
        self.today_trades = adjustments
        
        return transaction_cost
    
    def _update_portfolio_value(self):
        """Update portfolio value based on market moves and costs."""
        self.portfolio_value -= self.transaction_cost_total
        self.transaction_cost_total = 0
        
        old_portfolio_value = self.portfolio_value
        
        if self.current_step + 1 < len(self.market_data):
            portfolio_return = 0.0
            valid_weights_sum = 0.0
            
            for symbol in self.current_stocks:
                if symbol in self.current_weights:
                    weight = self.current_weights[symbol]
                    try:
                        if symbol in self.all_stock_data and self.current_step < len(self.all_stock_data[symbol]) and self.current_step + 1 < len(self.all_stock_data[symbol]):
                            current_price = self.all_stock_data[symbol].iloc[self.current_step]['close']
                            next_price = self.all_stock_data[symbol].iloc[self.current_step + 1]['close']
                            
                            if np.isnan(current_price) or np.isnan(next_price) or current_price <= 0:
                                continue
                            
                            stock_return = (next_price / current_price) - 1
                            stock_return = np.clip(stock_return, -0.1, 0.1)
                            
                            portfolio_return += weight * stock_return
                            valid_weights_sum += weight
                    except Exception as e:
                        print(f"Error computing return for {symbol}: {e}")
                        continue
            
            if valid_weights_sum > 0:
                portfolio_return = portfolio_return / valid_weights_sum
            else:
                portfolio_return = 0.0
            
            new_portfolio_value = old_portfolio_value * (1 + portfolio_return)
            
            if np.isnan(new_portfolio_value) or np.isinf(new_portfolio_value):
                print(f"Warning: invalid portfolio value {new_portfolio_value}; using previous value")
                new_portfolio_value = old_portfolio_value
            
            self.portfolio_value = max(new_portfolio_value, 0.01 * self.initial_capital)
        
        if self.portfolio_value > self.max_portfolio_value:
            self.max_portfolio_value = self.portfolio_value
        
        if self.max_portfolio_value > 0:
            self.current_drawdown = (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value
            self.current_drawdown = np.clip(self.current_drawdown, 0.0, 0.99)
    
    def _calculate_reward(self):
        """Calculate reward using returns and penalties."""
        try:
            if not hasattr(self, 'previous_portfolio_value') or self.previous_portfolio_value <= 0:
                print("Warning: previous portfolio value invalid; using 0 return")
                returns = 0.0
            else:
                returns = (self.portfolio_value / self.previous_portfolio_value) - 1
                
                returns = np.clip(returns, -0.5, 0.5)
            
            variance_penalty = self._calculate_portfolio_variance()
            
            turnover_penalty = 0.0
            if hasattr(self, 'previous_weights') and self.previous_weights:
                all_symbols = set(list(self.current_weights.keys()) + list(self.previous_weights.keys()))
                for symbol in all_symbols:
                    prev_weight = self.previous_weights.get(symbol, 0.0)
                    current_weight = self.current_weights.get(symbol, 0.0)
                    turnover_penalty += abs(current_weight - prev_weight)
                turnover_penalty = turnover_penalty / 2.0
            
            drawdown_penalty = self._calculate_drawdown()
            
            if np.isnan(variance_penalty) or np.isinf(variance_penalty):
                print(f"Warning: variance penalty invalid: {variance_penalty}; set to 0")
                variance_penalty = 0.0
            
            if np.isnan(turnover_penalty) or np.isinf(turnover_penalty):
                print(f"Warning: turnover penalty invalid: {turnover_penalty}; set to 0")
                turnover_penalty = 0.0
            
            if np.isnan(drawdown_penalty) or np.isinf(drawdown_penalty):
                print(f"Warning: drawdown penalty invalid: {drawdown_penalty}; set to 0")
                drawdown_penalty = 0.0
            
            lambda_1 = self.reward_params.get('lambda_1', 0.5)
            lambda_2 = self.reward_params.get('lambda_2', 0.2)
            lambda_3 = self.reward_params.get('lambda_3', 1.0)
            
            lambda_1 = max(0, lambda_1)
            lambda_2 = max(0, lambda_2)
            lambda_3 = max(0, lambda_3)
            
            reward = returns - \
                    lambda_1 * variance_penalty - \
                    lambda_2 * turnover_penalty - \
                    lambda_3 * drawdown_penalty
            
            if np.isnan(reward) or np.isinf(reward):
                print(f"Warning: invalid reward {reward}; set to 0")
                reward = 0.0
            
            reward = np.clip(reward, -10.0, 10.0)
            
            return reward
            
        except Exception as e:
            print(f"Error computing reward: {e}")
            return 0.0
    
    def _calculate_portfolio_variance(self):
        """Calculate portfolio variance using covariance matrix."""
        try:
            returns_data = []
            stock_symbols = []
            
            for symbol in self.current_stocks:
                if symbol in self.all_stock_data and symbol in self.current_weights:
                    df = self.all_stock_data[symbol]
                    window_size = min(30, self.current_step)
                    start_idx = max(0, self.current_step - window_size)
                    end_idx = self.current_step + 1
                    
                    if start_idx < len(df) and end_idx <= len(df):
                        returns_series = df.iloc[start_idx:end_idx]['daily_return'].values
                        
                        if np.isnan(returns_series).any():
                            returns_series = np.nan_to_num(returns_series, nan=0.0)
                        
                        returns_series = np.clip(returns_series, -0.2, 0.2)
                        
                        returns_data.append(returns_series)
                        stock_symbols.append(symbol)
            
            if len(returns_data) < 2 or len(stock_symbols) < 2:
                return 0
            
            returns_matrix = np.array(returns_data)
            cov_matrix = np.cov(returns_matrix)
            
            if np.isnan(cov_matrix).any() or np.isinf(cov_matrix).any():
                print("Warning: covariance matrix invalid; returning 0")
                return 0
            
            weights = np.array([self.current_weights[symbol] for symbol in stock_symbols])
            
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)
            else:
                weights = np.ones(len(weights)) / len(weights)
            
            portfolio_variance = weights.T @ cov_matrix @ weights
            
            if np.isnan(portfolio_variance) or np.isinf(portfolio_variance) or portfolio_variance < 0:
                print(f"Warning: invalid portfolio variance {portfolio_variance}; returning 0")
                return 0
            
            return min(portfolio_variance, 0.1)
            
        except Exception as e:
            print(f"Error computing portfolio variance: {e}")
            return 0
    
    def _calculate_drawdown(self):
        """Calculate current drawdown."""
        drawdown = (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value
        
        return max(0, drawdown)
    
    def _check_stop_loss(self):
        """Check whether stop-loss is triggered."""
        self.current_drawdown = (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value
        self.current_drawdown = np.clip(self.current_drawdown, 0.0, 0.99)
        
        effective_threshold = min(self.stop_loss_threshold * (1 + 0.2 * self.stop_loss_count), 0.5)
        
        if self.current_drawdown > effective_threshold:
            print(f"Stop-loss triggered: drawdown {self.current_drawdown:.2%} > threshold {effective_threshold:.2%}")
            self.stop_loss_triggered = True
            self.stop_loss_count += 1
            
            self._reset_after_stop_loss()
            
            return True
        
        self.stop_loss_triggered = False
        return False
    
    def _reset_after_stop_loss(self):
        """Reset portfolio state after stop-loss trigger."""
        self.max_portfolio_value = max(self.portfolio_value * 1.2, self.initial_capital * 0.5)
        
        self.current_drawdown = (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value
        
        previous_cash = self.cash
        self.cash = max(self.cash, self.portfolio_value * 0.3)
        
        print(f"Post stop-loss reset: max value {self.max_portfolio_value:.2f}, drawdown {self.current_drawdown:.2%}")
        print(f"Cash increased: {previous_cash:.2f} -> {self.cash:.2f} ({self.cash/self.portfolio_value:.2%})")
        
        if self.stop_loss_count % 3 == 0:
            print(f"Stop-loss triggered {self.stop_loss_count} times; resetting counter")
            self.stop_loss_count = 0
    
    def _check_rebalance_trigger(self):
        """Check whether to trigger a rebalance."""
        current_date = pd.to_datetime(self.market_data.iloc[self.current_step]['date'])
        
        if current_date.day == 1:
            print(f"Periodic rebalance triggered: {current_date}")
            self.rebalance_count += 1
            return True
            
        if hasattr(self, 'lim_calculator') and self.lim_calculator is not None:
            try:
                current_date_str = self.market_data.iloc[self.current_step]['date']
                top_stocks = self.lim_calculator.get_top_stocks(
                    self.data_loader.all_stock_data,
                    self.data_loader.benchmark['daily_return'],
                    current_date_str,
                    top_n=self.portfolio_size * 2
                )
                
                if not top_stocks:
                    print("LIM returned no stocks; skipping rebalance check")
                    return False
                
                stocks_to_replace = [s for s in self.current_stocks if s not in top_stocks]
                
                if len(stocks_to_replace) > len(self.current_stocks) / 2:
                    print(f"LIM-based rebalance triggered: replacing {len(stocks_to_replace)} stocks")
                    self.rebalance_count += 1
                    return True
            except Exception as e:
                print(f"Error during rebalance check: {e}")
                return False
                
        return False
    
    def _get_market_features(self, current_date):
        """Extract market-level features."""
        market_row = self.market_data.iloc[self.current_step]
        
        market_features = []
        
        market_features.append(market_row.get('daily_return', 0))
        market_features.append(market_row.get('volatility', 0))
        market_features.append(market_row.get('momentum', 0))
        
        if 'sentiment' in market_row:
            market_features.append(market_row['sentiment'])
        
        if 'trend' in market_row:
            market_features.append(market_row['trend'])
            
        return np.array(market_features, dtype=np.float32)
        
    def _get_stock_features(self, current_date):
        """Extract per-stock features for the current portfolio."""
        all_stock_features = []
        
        for symbol in self.current_stocks:
            if symbol not in self.current_stock_data:
                stock_features = np.zeros(10)
            else:
                stock_data = self.current_stock_data[symbol]
                
                stock_df = stock_data[stock_data['date'] <= current_date]
                
                if len(stock_df) == 0:
                    stock_features = np.zeros(10)
                else:
                    stock_row = stock_df.iloc[-1]
                    
                    stock_features = []
                    
                    stock_features.append(stock_row.get('daily_return', 0))
                    
                    stock_features.append(stock_row.get('volatility', 0))
                    
                    if 'pe_ratio' in stock_row:
                        stock_features.append(stock_row['pe_ratio'])
                        
                    if 'rsi_14' in stock_row:
                        stock_features.append(stock_row['rsi_14'])
                    
                    if 'macd' in stock_row:
                        stock_features.append(stock_row['macd'])
                    
                    if 'bb_width' in stock_row:
                        stock_features.append(stock_row['bb_width'])
                    
                    if 'atr_14' in stock_row:
                        stock_features.append(stock_row['atr_14'])
                    
                    if 'relative_strength' in stock_row:
                        stock_features.append(stock_row['relative_strength'])
                    
                    if 'volume_change' in stock_row:
                        stock_features.append(stock_row['volume_change'])
            
            all_stock_features.extend(stock_features)
            
        return np.array(all_stock_features, dtype=np.float32)
    
    def _get_portfolio_features(self):
        """Extract portfolio-level features."""
        portfolio_features = []
        
        portfolio_features.append(self.portfolio_value / self.initial_capital)
        
        portfolio_features.append(self.current_drawdown)
        
        if hasattr(self, 'previous_portfolio_value') and self.previous_portfolio_value > 0:
            portfolio_change = (self.portfolio_value / self.previous_portfolio_value) - 1
        else:
            portfolio_change = 0
        portfolio_features.append(portfolio_change)
        
        cash_ratio = self.cash / self.portfolio_value if self.portfolio_value > 0 else 0
        portfolio_features.append(cash_ratio)
        
        portfolio_features.append(1.0 if self.stop_loss_triggered else 0.0)
        
        return np.array(portfolio_features, dtype=np.float32)
        
    def _record_state(self, info):
        """Record the current state to history."""
        current_date = self.market_data.iloc[self.current_step]['date']
        
        state_record = {
            'date': current_date,
            'portfolio_value': self.portfolio_value,
            'weights': self.current_weights.copy(),
            'stocks': self.current_stocks.copy(),
            'transaction_cost': info.get('transaction_cost', 0),
            'stop_loss': info.get('stop_loss_triggered', False),
            'rebalance': info.get('rebalance_triggered', False)
        }
        
        self.history.append(state_record)
        
        self.previous_portfolio_value = self.portfolio_value 

