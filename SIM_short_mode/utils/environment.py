"""
Trading environment module based on OpenAI Gym.
"""

import numpy as np
import pandas as pd
import gym
from gym import spaces
from sklearn.decomposition import PCA
import warnings
import os
from datetime import datetime

warnings.filterwarnings("ignore", category=RuntimeWarning)

class TradingEnvironment(gym.Env):
    """Trading environment for reinforcement learning."""
    
    def __init__(self, config, data_loader, sim_calculator):
        """Initialize the trading environment."""
        super(TradingEnvironment, self).__init__()
        
        self.config = config
        self.data_loader = data_loader
        self.sim_calculator = sim_calculator
        
        self.portfolio_size = config['environment']['portfolio_size']
        self.initial_capital = config['environment']['initial_capital']
        self.commission_rate = config['environment']['commission_rate']
        self.group_type = config['environment'].get('group_type', 'TOP')
        
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
        self.shares = {}
        
        self.short_holdings = {}
        self.short_shares = {}
        
        self.current_stocks = []
        self.previous_portfolio_value = self.initial_capital
        
        self.returns_history = []
        self.portfolio_values = []
        self.cumulative_returns = []
        self.max_return = 0.0
        
        self.rebalance_count = 0
        self.stop_loss_count = 0
        self.current_drawdown = 0
        self.transaction_cost_total = 0
        
        self.history = []
        
        self.stop_loss_triggered = False
        self.today_trades = {}
        
        self.state_buffer = []
        self.buffer_size = 50
        self.pca_fitted = False
        self.pca_model = None
        
        self.atr_window = config.get('environment', {}).get('atr_window', 14)
        self.volatility_range = config.get('environment', {}).get('volatility_range', (0.5, 2.0))
        self.use_trend_adjustment = config.get('environment', {}).get('trend_adjustment', True)
        self.atr_history = []
        
    def _calculate_state_dim(self):
        """Calculate state vector dimension."""
        market_feature_count = 3
        stock_feature_count = 2
        portfolio_feature_count = 5
        
        total_dim = market_feature_count + (stock_feature_count * self.portfolio_size) + portfolio_feature_count
        
        print(f"Calculated state dimension: {total_dim}")
        return total_dim
    
    def reset(self, start_date=None, end_date=None):
        """Reset environment state."""
        if start_date is None:
            start_date = self.config['data']['start_date']
        if end_date is None:
            end_date = self.config['data']['end_date']
            
        self.start_date_str = start_date
        self.end_date_str = end_date
            
        print(f"Loading data from {start_date} to {end_date}...")
        
        self.data_loader.set_date_range(start_date, end_date)
        
        market_data = self.data_loader.benchmark
        if isinstance(market_data, pd.DataFrame) and not market_data.empty and 'date' in market_data.columns:
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            market_data['date'] = pd.to_datetime(market_data['date'])
            market_data = market_data[(market_data['date'] >= start) & (market_data['date'] <= end)].reset_index(drop=True)
            self.data_loader.benchmark = market_data
            if not market_data.empty:
                min_date = market_data['date'].min().date()
                max_date = market_data['date'].max().date()
                print(f"Effective market date range: {min_date} to {max_date}")
        
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
        self.shares = {}
        
        self.short_holdings = {}
        self.short_shares = {}
        
        self.current_stocks = []
        self.stop_loss_triggered = False
        self.previous_portfolio_value = self.initial_capital
        
        self.returns_history = []
        self.portfolio_values = []
        self.cumulative_returns = []
        self.max_return = 0.0
        
        self.history = []
        
        self.rebalance_count = 0
        self.stop_loss_count = 0
        self.max_drawdown = 0
        self.current_drawdown = 0
        self.transaction_cost_total = 0
        
        self.state_buffer = []
        self.pca_fitted = False
        self.pca_model = None
        
        self._select_portfolio_stocks(self.group_type)
        
        initial_state = self._get_state()
        
        return initial_state
    
    def step(self, action):
        """Execute one environment step."""
        if self.current_step >= len(self.market_data) - 1:
            print(
                f"Warning: current_step({self.current_step}) reached or exceeded "
                f"market_data max index ({len(self.market_data) - 1}). Ending early."
            )
            last_state = self._get_state()
            return last_state, 0, True, {'portfolio_value': self.portfolio_value, 'day': 'Final'}
            
        self.previous_weights = self.current_weights.copy() if self.current_weights else {}
        
        if len(action) != len(self.current_stocks):
            raise ValueError(
                f"Action dimension ({len(action)}) does not match number of portfolio stocks "
                f"({len(self.current_stocks)})"
            )
            
        if np.sum(action) > 0:
            action = action / np.sum(action)
        else:
            action = np.ones(len(action)) / len(action)
            
        target_weights = dict(zip(self.current_stocks, action))
        
        target_weights = self._apply_trading_constraints(target_weights)
        
        transaction_cost = self._execute_trades(target_weights)
        
        self.current_weights = target_weights
        
        self._ensure_valid_state()
        
        self._update_portfolio_value()
        
        stop_loss_triggered = self._check_stop_loss()
        
        rebalance_triggered = self._check_rebalance_trigger()
        
        if rebalance_triggered:
            self._select_portfolio_stocks(self.group_type)
        
        reward = self._calculate_reward()
        
        done = self.current_step >= len(self.market_data) - 2
        
        try:
            if isinstance(self.market_data, pd.DataFrame) and 'date' in self.market_data.columns:
                if 0 <= self.current_step < len(self.market_data):
                    current_day = self.market_data.iloc[self.current_step]['date']
                else:
                    current_day = f"Step_{self.current_step}"
            else:
                current_day = f"Step_{self.current_step}"
        except Exception as e:
            print(f"Error getting current date info: {e}")
            current_day = f"Step_{self.current_step}"
        
        info = {
            'portfolio_value': self.portfolio_value,
            'weights': self.current_weights,
            'transaction_cost': transaction_cost,
            'stop_loss_triggered': stop_loss_triggered,
            'rebalance_triggered': rebalance_triggered,
            'current_stocks': self.current_stocks,
            'day': current_day
        }
        
        self._record_state(info)
        
        self.current_step += 1
        
        next_state = self._get_state()
        
        self._ensure_valid_state()
        
        return next_state, reward, done, info
    
    def _get_state(self):
        """Build the state vector."""
        if self.current_step >= len(self.market_data):
            safe_step = len(self.market_data) - 1
            print(
                f"Warning: current_step({self.current_step}) exceeds market_data range "
                f"({len(self.market_data)}), using last valid index ({safe_step})"
            )
            current_date = self.market_data.iloc[safe_step]['date']
        else:
            current_date = self.market_data.iloc[self.current_step]['date']
        
        market_features = self._get_market_features(current_date)
        
        stock_features = self._get_stock_features(current_date)
        
        portfolio_features = self._get_portfolio_features()
        
        full_state = np.concatenate([market_features, stock_features, portfolio_features])
        
        use_pca = self.config.get('features', {}).get('use_pca', False)
        if use_pca:
            n_components = self.config.get('features', {}).get('pca_components', 20)
            
            try:
                feature_dim = len(full_state)
                max_allowed = max(1, int(feature_dim * 0.8))
                n_components = min(n_components, max_allowed)
                
                synthetic_samples = 50
                
                noise_scale = 0.001
                synthetic_data = np.tile(full_state, (synthetic_samples, 1))
                synthetic_data += np.random.normal(0, noise_scale, synthetic_data.shape)
                
                pca = PCA(n_components=n_components)
                state = pca.fit_transform([full_state])[0]
                
                if self.current_step == 0 or self.current_step % 100 == 0:
                    explained_variance_ratio = pca.explained_variance_ratio_
                    cumulative_variance = np.cumsum(explained_variance_ratio)
                    print(
                        f"PCA succeeded: {feature_dim} -> {n_components}, "
                        f"variance explained: {cumulative_variance[-1]:.2f}"
                    )
                    
            except Exception as e:
                print(f"PCA failed: {e}. Using raw features.")
                state = full_state
        else:
            state = full_state
        
        actual_dim = len(state)
        expected_dim = self.observation_space.shape[0]
        
        if actual_dim != expected_dim:
            if actual_dim > expected_dim:
                state = state[:expected_dim]
            else:
                padding = np.zeros(expected_dim - actual_dim, dtype=np.float32)
                state = np.concatenate([state, padding])
                
        return state
    
    def _select_portfolio_stocks(self, group_type='TOP'):
        """Select portfolio stocks based on SIM*."""
        try:
            current_date = self.market_data.iloc[self.current_step]['date']
            
            all_stock_data = self.data_loader.all_stock_data
            benchmark_returns = self.data_loader.benchmark['daily_return']
            
            if group_type == 'TOP':
                selected_stocks = self.sim_calculator.get_top_stocks(
                    all_stock_data, 
                    benchmark_returns, 
                    current_date, 
                    self.portfolio_size
                )
                print(f"SIM* selected short stocks: {selected_stocks}")
            elif group_type == 'MIDDLE':
                sorted_stocks = self.sim_calculator.get_sorted_stocks(all_stock_data, benchmark_returns, current_date)
                if sorted_stocks:
                    n = len(sorted_stocks)
                    middle_start = n // 3
                    middle_end = 2 * n // 3
                    selected_stocks = [symbol for symbol, _ in sorted_stocks[middle_start:middle_end]]
                    selected_stocks = selected_stocks[:self.portfolio_size]
                    print(f"Middle group short stocks: {selected_stocks}")
                else:
                    selected_stocks = []
            elif group_type == 'LOW':
                sorted_stocks = self.sim_calculator.get_sorted_stocks(all_stock_data, benchmark_returns, current_date)
                if sorted_stocks:
                    selected_stocks = [symbol for symbol, _ in sorted_stocks[-self.portfolio_size:]]
                    print(f"Bottom group short stocks: {selected_stocks}")
                else:
                    selected_stocks = []
            else:
                selected_stocks = self.sim_calculator.get_top_stocks(
                    all_stock_data, 
                    benchmark_returns, 
                    current_date, 
                    self.portfolio_size
                )
                
            if len(selected_stocks) < self.portfolio_size:
                print(f"Warning: insufficient selected stocks ({len(selected_stocks)}/{self.portfolio_size})")
                
                if not selected_stocks:
                    avaSIMble_stocks = list(self.all_stock_data.keys())
                    avaSIMble_stocks = [s for s in avaSIMble_stocks if s in all_stock_data]
                    
                    if len(avaSIMble_stocks) > self.portfolio_size:
                        import random
                        random.shuffle(avaSIMble_stocks)
                        selected_stocks = avaSIMble_stocks[:self.portfolio_size]
                    else:
                        selected_stocks = avaSIMble_stocks
                    
                    print(f"Using available stocks: {selected_stocks}")
            
            self.current_stocks = selected_stocks
            
            self.current_stock_data = {}
            for symbol in self.current_stocks:
                if symbol in all_stock_data:
                    self.current_stock_data[symbol] = all_stock_data[symbol]
            
            if selected_stocks:
                equal_weight = 1.0 / len(selected_stocks)
                self.current_weights = {symbol: equal_weight for symbol in selected_stocks}
            else:
                self.current_weights = {}
                
        except Exception as e:
            print(f"Error selecting portfolio stocks: {e}")
            import traceback
            traceback.print_exc()
            
            avaSIMble_stocks = list(self.all_stock_data.keys())
            if len(avaSIMble_stocks) > self.portfolio_size:
                import random
                random.shuffle(avaSIMble_stocks)
                self.current_stocks = avaSIMble_stocks[:self.portfolio_size]
            else:
                self.current_stocks = avaSIMble_stocks
                
            if self.current_stocks:
                equal_weight = 1.0 / len(self.current_stocks)
                self.current_weights = {symbol: equal_weight for symbol in self.current_stocks}
            else:
                self.current_weights = {}
    
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
        """Execute trades for a short-only portfolio."""
        current_date = self.market_data.iloc[self.current_step]['date']
        current_timestamp = pd.Timestamp(current_date)
        
        trade_records = []
        
        if not hasattr(self, 'short_holdings') or self.short_holdings is None:
            self.short_holdings = {}
            
        total_value = self.portfolio_value
        
        current_prices = {}
        for symbol in set(list(self.current_stocks) + list(self.short_holdings.keys())):
            try:
                if symbol in self.all_stock_data and self.current_step < len(self.all_stock_data[symbol]):
                    current_prices[symbol] = self.all_stock_data[symbol].iloc[self.current_step]['close']
            except Exception as e:
                print(f"Error getting price for {symbol}: {e}")
        
        if not hasattr(self, 'short_shares') or self.short_shares is None:
            self.short_shares = {}
            for symbol, value in self.short_holdings.items():
                if symbol in current_prices and current_prices[symbol] > 0:
                    self.short_shares[symbol] = int(value / current_prices[symbol])
        
        target_values = {}
        for symbol, weight in target_weights.items():
            target_values[symbol] = total_value * weight
            
        adjustments = {}
        for symbol in set(list(self.short_shares.keys()) + list(target_weights.keys())):
            current_short_shares = self.short_shares.get(symbol, 0)
            current_value = current_short_shares * current_prices.get(symbol, 0) if symbol in current_prices else 0
            target_value = target_values.get(symbol, 0)
            adjustments[symbol] = target_value - current_value
        
        for symbol in list(self.short_shares.keys()):
            if symbol not in target_weights or target_weights[symbol] == 0:
                if symbol in current_prices and current_prices[symbol] > 0:
                    shares_to_cover = self.short_shares[symbol]
                    if shares_to_cover > 0:
                        cover_value = shares_to_cover * current_prices[symbol]
                        print(
                            f"[{current_timestamp}] Cover {symbol}: buy {shares_to_cover} shares "
                            f"at {current_prices[symbol]:.2f}, value {cover_value:.2f}"
                        )
                        self.cash -= cover_value * (1 + self.commission_rate)
                        del self.short_shares[symbol]
                        
                        trade_records.append({
                            'timestamp': current_timestamp,
                            'type': 'cover_short',
                            'symbol': symbol,
                            'shares': shares_to_cover,
                            'price': current_prices[symbol],
                            'value': cover_value,
                            'reason': 'portfolio_adjustment'
                        })
        
        cover_adjustments = {symbol: adj for symbol, adj in adjustments.items() if adj < 0}
        new_short_adjustments = {symbol: adj for symbol, adj in adjustments.items() if adj > 0}
        
        total_cover = 0
        cover_cost = 0
        for symbol, adjustment in cover_adjustments.items():
            if symbol in current_prices and current_prices[symbol] > 0:
                shares_to_cover = int(abs(adjustment) / current_prices[symbol])
                current_short_shares = self.short_shares.get(symbol, 0)
                shares_to_cover = min(shares_to_cover, current_short_shares)
                
                if shares_to_cover > 0:
                    cover_value = shares_to_cover * current_prices[symbol]
                    total_cover += cover_value
                    
                    self.short_shares[symbol] = current_short_shares - shares_to_cover
                    if self.short_shares[symbol] <= 0:
                        del self.short_shares[symbol]
                    
                    print(
                        f"[{current_timestamp}] Cover {symbol}: buy {shares_to_cover} shares "
                        f"at {current_prices[symbol]:.2f}, value {cover_value:.2f}"
                    )
                    
                    trade_records.append({
                        'timestamp': current_timestamp,
                        'type': 'cover_short',
                        'symbol': symbol,
                        'shares': shares_to_cover,
                        'price': current_prices[symbol],
                        'value': cover_value,
                        'reason': 'rebalance'
                    })
        
        cover_cost = total_cover * self.commission_rate
        
        self.cash = self.cash - total_cover - cover_cost
        
        avaSIMble_cash = self.cash * 0.95
        
        total_short_desired = 0
        shares_to_short_dict = {}
        
        for symbol, adjustment in new_short_adjustments.items():
            if symbol in current_prices and current_prices[symbol] > 0:
                target_shares = int(adjustment / current_prices[symbol])
                short_value = target_shares * current_prices[symbol]
                total_short_desired += short_value
                shares_to_short_dict[symbol] = target_shares
        
        margin_requirement = total_short_desired * 0.5
        estimated_short_cost = total_short_desired * self.commission_rate
        if margin_requirement + estimated_short_cost > avaSIMble_cash:
            reduction_ratio = avaSIMble_cash / (margin_requirement + estimated_short_cost)
            print(
                f"Warning: reducing short exposure. Available cash: {avaSIMble_cash:.2f}, "
                f"required margin: {margin_requirement:.2f}, reduction: {reduction_ratio:.2%}"
            )
            
            for symbol in shares_to_short_dict:
                shares_to_short_dict[symbol] = max(1, int(shares_to_short_dict[symbol] * reduction_ratio))
        
        total_short = 0
        short_cost = 0
        
        for symbol, adjustment in new_short_adjustments.items():
            if symbol in current_prices and current_prices[symbol] > 0:
                shares_to_short = shares_to_short_dict.get(symbol, 0)
                
                if shares_to_short > 0:
                    short_value = shares_to_short * current_prices[symbol]
                    margin_needed = short_value * 0.5
                    short_cost_estimate = short_value * self.commission_rate
                    
                    if margin_needed + short_cost_estimate <= self.cash:
                        total_short += short_value
                        
                        self.short_shares[symbol] = self.short_shares.get(symbol, 0) + shares_to_short
                        
                        print(
                            f"[{current_timestamp}] Short {symbol}: {shares_to_short} shares "
                            f"at {current_prices[symbol]:.2f}, value {short_value:.2f}"
                        )
                        
                        self.cash += short_value * (1 - self.commission_rate)
                        
                        trade_records.append({
                            'timestamp': current_timestamp,
                            'type': 'short',
                            'symbol': symbol,
                            'shares': shares_to_short,
                            'price': current_prices[symbol],
                            'value': short_value,
                            'reason': 'rebalance'
                        })
                    else:
                        print(
                            f"Warning: skipping short {symbol}; insufficient margin. "
                            f"Needed {margin_needed + short_cost_estimate:.2f}, remaining {self.cash:.2f}"
                        )
        
        short_cost = total_short * self.commission_rate
        
        self.holdings = {}
        self.short_holdings = {}
        for symbol, shares in self.short_shares.items():
            if symbol in current_prices:
                self.short_holdings[symbol] = shares * current_prices[symbol]
        
        transaction_cost = cover_cost + short_cost
        
        self.transaction_cost_total += transaction_cost
        
        self.today_trades = {
            'timestamp': current_timestamp,
            'trades': trade_records,
            'adjustments': adjustments,
            'transaction_cost': transaction_cost
        }
        
        short_holdings_value = sum(self.short_holdings.values()) if self.short_holdings else 0.0
        
        calculated_portfolio_value = self.cash - short_holdings_value
        
        if abs(calculated_portfolio_value - self.portfolio_value) > 1.0:
            print(
                f"Warning: portfolio value mismatch. Updating: {self.portfolio_value:.2f} "
                f"-> {calculated_portfolio_value:.2f}"
            )
            print(f"Short holdings value: {short_holdings_value:.2f}, cash: {self.cash:.2f}")
            self.portfolio_value = calculated_portfolio_value
        
        if trade_records:
            print(f"\n[{current_timestamp}] Trade summary:")
            print(f"Total cover amount: {total_cover:.2f}")
            print(f"Total new short amount: {total_short:.2f}")
            print(f"Transaction cost: {transaction_cost:.2f}")
            print(f"Cash: {self.cash:.2f}")
            print(f"Short holdings value: {short_holdings_value:.2f}")
            print(f"Portfolio value: {self.portfolio_value:.2f}\n")
        
        return transaction_cost
    
    def _check_margin_requirements(self):
        """Check whether margin requirements are satisfied."""
        if not self.short_holdings:
            return True
            
        short_value = 0
        for symbol, shares in self.short_shares.items():
            if symbol in self.all_stock_data and self.current_step < len(self.all_stock_data[symbol]):
                current_price = self.all_stock_data[symbol].iloc[self.current_step]['close']
                short_value += shares * current_price
        
        required_margin = short_value * 0.5
        
        if self.cash < required_margin:
            print(f"Warning: insufficient margin. Required: {required_margin:.2f}, cash: {self.cash:.2f}")
            return False
            
        return True
        
    def _update_portfolio_value(self):
        """Update portfolio value for short-only strategy."""
        self.portfolio_value -= self.transaction_cost_total
        self.transaction_cost_total = 0
        
        old_portfolio_value = self.portfolio_value
        
        margin_sufficient = self._check_margin_requirements()
        if not margin_sufficient:
            print("Warning: insufficient margin. Forced liquidation may be required.")
        
        if self.current_step + 1 < len(self.market_data):
            portfolio_return = 0.0
            valid_weights_sum = 0.0
            
            self.short_holdings = {}
            current_prices = {}
            next_prices = {}
            
            for symbol in self.short_shares.keys():
                if symbol in self.all_stock_data and self.current_step < len(self.all_stock_data[symbol]):
                    current_prices[symbol] = self.all_stock_data[symbol].iloc[self.current_step]['close']
                    if self.current_step + 1 < len(self.all_stock_data[symbol]):
                        next_prices[symbol] = self.all_stock_data[symbol].iloc[self.current_step + 1]['close']
                        
                    shares = self.short_shares[symbol]
                    self.short_holdings[symbol] = shares * current_prices[symbol]
            
            for symbol, shares in self.short_shares.items():
                if symbol in current_prices and symbol in next_prices and current_prices[symbol] > 0:
                    weight = (shares * current_prices[symbol]) / self.portfolio_value
                    price_return = (current_prices[symbol] - next_prices[symbol]) / current_prices[symbol]
                    weighted_return = weight * price_return
                    portfolio_return += weighted_return
                    valid_weights_sum += weight
            
            if valid_weights_sum > 0:
                portfolio_return = portfolio_return * valid_weights_sum
            else:
                portfolio_return = 0.0
            
            self.portfolio_value = old_portfolio_value * (1 + portfolio_return)
            
            for symbol, shares in self.short_shares.items():
                if symbol in next_prices:
                    self.short_holdings[symbol] = shares * next_prices[symbol]
            
            self.returns_history.append(portfolio_return)
            self.portfolio_values.append(self.portfolio_value)
            
            if len(self.returns_history) == 1:
                self.cumulative_returns.append(portfolio_return)
            else:
                self.cumulative_returns.append((1 + self.cumulative_returns[-1]) * (1 + portfolio_return) - 1)
            
            if self.cumulative_returns and self.cumulative_returns[-1] > self.max_return:
                self.max_return = self.cumulative_returns[-1]
            
            self._calculate_drawdown()
        
        return old_portfolio_value
    
    def _calculate_reward(self):
        """Compute reward for short-only strategy."""
        try:
            if not hasattr(self, 'previous_portfolio_value') or self.previous_portfolio_value <= 0:
                print("Warning: invalid previous portfolio value, using 0 return")
                returns = 0.0
            else:
                returns = (self.portfolio_value / self.previous_portfolio_value) - 1
                
                returns = np.clip(returns, -0.5, 0.5)
            
            initial_capital_return = (self.portfolio_value / self.initial_capital) - 1
            
            vertical_reward = 0.0
            
            if initial_capital_return > 0:
                positive_tier = int(initial_capital_return / 0.1)
                vertical_reward = positive_tier * 2.0
            else:
                negative_tier = int(abs(initial_capital_return) / 0.05)
                vertical_reward = -negative_tier * 0.5
            
            if not hasattr(self, 'consecutive_up_days'):
                self.consecutive_up_days = 0
                
            if returns > 0:
                self.consecutive_up_days += 1
            else:
                self.consecutive_up_days = 0
                
            horizontal_reward = 0.0
            if self.consecutive_up_days >= 2:
                horizontal_reward = min(self.consecutive_up_days * 0.5, 4.0)
            
            end_period_reward = 0.0
            
            is_last_day = self.current_step == len(self.market_data) - 1
            
            if is_last_day and self.portfolio_value > self.initial_capital:
                final_return_pct = (self.portfolio_value / self.initial_capital - 1) * 100
                end_period_reward = 2.0 if final_return_pct > 0 else 0.0
            
            base_reward = returns * 3.0
            
            reward = base_reward + vertical_reward + horizontal_reward + end_period_reward
            
            if hasattr(self, 'history') and isinstance(self.history, list) and len(self.history) > 0:
                current_record = self.history[-1]
                
                current_record['returns'] = returns
                current_record['vertical_reward'] = vertical_reward
                current_record['horizontal_reward'] = horizontal_reward  
                current_record['end_period_reward'] = end_period_reward
                current_record['base_reward'] = base_reward
                current_record['total_reward'] = reward
            
            if np.isnan(reward) or np.isinf(reward):
                print(f"Warning: invalid reward {reward}, set to 0")
                reward = 0.0
            
            reward = np.clip(reward, -7.0, 14.0)
            
            if self.current_step % 20 == 0 or is_last_day:
                print(
                    f"Reward diagnostics - step {self.current_step}: return={returns:.4f}, "
                    f"reward={reward:.4f} (base={base_reward:.2f}, vertical={vertical_reward:.2f}, "
                    f"horizontal={horizontal_reward:.2f}, end={end_period_reward:.2f})"
                )
                print(
                    f"Portfolio value: {self.portfolio_value:.2f}, "
                    f"return vs initial: {initial_capital_return:.2%}"
                )
            
            return reward
            
        except Exception as e:
            print(f"Error computing reward: {e}")
            return 0.0
    
    def _calculate_portfolio_variance(self):
        """Compute portfolio variance from covariance matrix."""
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
                print("Warning: covariance matrix invalid, returning 0")
                return 0
            
            weights = np.array([self.current_weights[symbol] for symbol in stock_symbols])
            
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)
            else:
                weights = np.ones(len(weights)) / len(weights)
            
            portfolio_variance = weights.T @ cov_matrix @ weights
            
            if np.isnan(portfolio_variance) or np.isinf(portfolio_variance) or portfolio_variance < 0:
                print(f"Warning: invalid portfolio variance {portfolio_variance}, returning 0")
                return 0
            
            return min(portfolio_variance, 0.1)
            
        except Exception as e:
            print(f"Error computing portfolio variance: {e}")
            return 0
    
    def _calculate_drawdown(self):
        """Compute current drawdown."""
        drawdown = (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value
        
        return max(0, drawdown)
    
    def _check_stop_loss(self):
        """Check stop-loss (disabled)."""
        return False
    
    def _reset_after_stop_loss(self):
        """Reset portfolio state after stop-loss."""
        self.max_portfolio_value = max(self.portfolio_value * 1.2, self.initial_capital * 0.5)
        
        self.current_drawdown = (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value
        
        previous_cash = self.cash
        
        current_short_holdings = {}
        if hasattr(self, 'short_holdings') and self.short_holdings:
            current_short_holdings = self.short_holdings.copy()
        
        current_short_shares = {}
        if hasattr(self, 'short_shares') and self.short_shares:
            current_short_shares = self.short_shares.copy()
        
        total_short_value = sum(current_short_holdings.values()) if current_short_holdings else 0
        
        if total_short_value > 0:
            target_short_ratio = 0.3
            target_short_value = self.portfolio_value * target_short_ratio
            
            if total_short_value > target_short_value:
                reduction_ratio = target_short_value / total_short_value
                print(f"Reducing short exposure after stop-loss by {reduction_ratio:.2%}")
                
                for symbol in list(current_short_shares.keys()):
                    shares_to_cover = int(current_short_shares[symbol] * (1 - reduction_ratio))
                    if shares_to_cover > 0:
                        current_price = 0
                        if symbol in self.all_stock_data and self.current_step < len(self.all_stock_data[symbol]):
                            current_price = self.all_stock_data[symbol].iloc[self.current_step]['close']
                        
                        if current_price > 0:
                            cover_value = shares_to_cover * current_price
                            
                            self.cash -= cover_value * (1 + self.commission_rate)
                            
                            current_short_shares[symbol] -= shares_to_cover
                            if current_short_shares[symbol] <= 0:
                                del current_short_shares[symbol]
                                if symbol in current_short_holdings:
                                    del current_short_holdings[symbol]
                            elif symbol in current_short_holdings:
                                current_short_holdings[symbol] = current_short_shares[symbol] * current_price
                
                self.short_shares = current_short_shares
                self.short_holdings = current_short_holdings
                
                cash_ratio = self.cash / self.portfolio_value
                short_value = sum(self.short_holdings.values()) if self.short_holdings else 0
                short_ratio = short_value / self.portfolio_value
                
                print(
                    f"Post stop-loss adjustment: cash {previous_cash:.2f} -> {self.cash:.2f} "
                    f"({cash_ratio:.2%}), short ratio {short_ratio:.2%}"
                )
        else:
            print(
                f"No short holdings to adjust. Cash unchanged: {self.cash:.2f} "
                f"({self.cash/self.portfolio_value:.2%})"
            )
        
        print(
            f"Post stop-loss reset: max value {self.max_portfolio_value:.2f}, "
            f"drawdown {self.current_drawdown:.2%}"
        )
        
        if self.stop_loss_count % 3 == 0:
            print(f"Stop-loss triggered {self.stop_loss_count} times. Resetting count.")
            self.stop_loss_count = 0
    
    def _check_rebalance_trigger(self):
        """Check whether monthly rebalance is triggered."""
        current_date = pd.to_datetime(self.market_data.iloc[self.current_step]['date'])
        
        if current_date.day <= 5:
            month_start = pd.Timestamp(year=current_date.year, month=current_date.month, day=1)
            month_trading_days = [pd.to_datetime(self.market_data.iloc[i]['date']) 
                                 for i in range(self.current_step+1) 
                                 if pd.to_datetime(self.market_data.iloc[i]['date']).month == current_date.month
                                 and pd.to_datetime(self.market_data.iloc[i]['date']).year == current_date.year]
            
            if month_trading_days and month_trading_days[0] == current_date:
                print(f"Month-start rebalance triggered: {current_date}")
                self.rebalance_count += 1
                return True
                
        return False
    
    def _get_market_features(self, current_date):
        """Extract market-level features."""
        if self.current_step >= len(self.market_data):
            safe_step = len(self.market_data) - 1
            market_row = self.market_data.iloc[safe_step]
        else:
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
        """Extract per-stock features for the portfolio."""
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
        
        cash_ratio = min(1.0, self.cash / self.portfolio_value) if self.portfolio_value > 0 else 0
        portfolio_features.append(cash_ratio)
        
        portfolio_features.append(1.0 if self.stop_loss_triggered else 0.0)
        
        return np.array(portfolio_features, dtype=np.float32)
        
    def _record_state(self, info):
        """Record current state in history."""
        try:
            if 'day' in info and info['day'] is not None:
                current_date = info['day']
            else:
                if 0 <= self.current_step < len(self.market_data):
                    if isinstance(self.market_data, pd.DataFrame):
                        current_date = self.market_data.iloc[self.current_step]['date'] if 'date' in self.market_data.columns else None
                    else:
                        current_date = f"Step_{self.current_step}"
                else:
                    current_date = f"Step_{self.current_step}"
        except Exception as e:
            print(f"Error getting current date: {e}")
            current_date = f"Step_{self.current_step}"
        
        cash_ratio = min(1.0, self.cash / self.portfolio_value) if self.portfolio_value > 0 else 0
        
        short_holdings_value = sum(self.short_holdings.values()) if hasattr(self, 'short_holdings') and self.short_holdings else 0.0
        short_ratio = min(1.0, short_holdings_value / self.portfolio_value) if self.portfolio_value > 0 else 0
        
        state_record = {
            'date': current_date,
            'portfolio_value': self.portfolio_value,
            'initial_capital': self.initial_capital,
            'return_pct': (self.portfolio_value / self.initial_capital - 1) * 100,
            'daily_return': (self.portfolio_value / self.previous_portfolio_value - 1)
            if hasattr(self, 'previous_portfolio_value') and self.previous_portfolio_value > 0 else 0,
            'weights': self.current_weights.copy(),
            'stocks': self.current_stocks.copy(),
            'cash': self.cash,
            'cash_ratio': cash_ratio,
            'short_ratio': short_ratio,
            'drawdown': self.current_drawdown * 100,
            'max_value': self.max_portfolio_value,
            'transaction_cost': info.get('transaction_cost', 0),
            'stop_loss': info.get('stop_loss_triggered', False),
            'rebalance': info.get('rebalance_triggered', False),
            'short_holdings': self.short_holdings.copy() if hasattr(self, 'short_holdings') and self.short_holdings else {},
            'short_shares': self.short_shares.copy() if hasattr(self, 'short_shares') and self.short_shares else {}
        }
        
        if hasattr(self, 'atr_history') and len(self.atr_history) > 0 and self.current_step < len(self.atr_history):
            state_record['atr'] = self.atr_history[self.current_step]
            state_record['stop_loss_threshold'] = self.stop_loss_threshold_history[self.current_step]
        else:
            state_record['atr'] = None
            state_record['stop_loss_threshold'] = self.original_stop_loss_threshold if hasattr(self, 'original_stop_loss_threshold') else None
        
        if not hasattr(self, 'history') or not isinstance(self.history, list):
            self.history = []
        self.history.append(state_record)
        
        self.previous_portfolio_value = self.portfolio_value
    
    def export_history(self, file_path=None):
        """Export trade history to CSV."""
        if not self.history:
            print("No history to export")
            return None
        
        basic_data = []
        for record in self.history:
            row = {
                'date': record['date'],
                'portfolio_value': record['portfolio_value'],
                'return_pct': record.get('return_pct', (record['portfolio_value'] / self.initial_capital - 1) * 100),
                'daily_return': record.get('daily_return', 0),
                'cash': record.get('cash', 0),
                'cash_ratio': record.get('cash_ratio', 0),
                'short_ratio': record.get('short_ratio', 0),
                'drawdown': record.get('drawdown', 0),
                'transaction_cost': record['transaction_cost'],
                'stop_loss': record['stop_loss'],
                'rebalance': record['rebalance'],
                'stock_count': len(record['stocks'])
            }
            basic_data.append(row)
        
        df_basic = pd.DataFrame(basic_data)
        
        if file_path is None:
            results_dir = 'results'
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(results_dir, f"portfolio_history_{timestamp}.csv")
        else:
            output_dir = os.path.dirname(file_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
        
        df_basic.to_csv(file_path, index=False)
        print(f"Basic portfolio history saved to: {file_path}")
        
        all_symbols = set()
        for record in self.history:
            all_symbols.update(record['weights'].keys())
            if 'short_holdings' in record and record['short_holdings']:
                all_symbols.update(record['short_holdings'].keys())
        
        holdings_data = []
        for record in self.history:
            row = {'date': record['date']}
            for symbol in all_symbols:
                row[f'weight_{symbol}'] = record['weights'].get(symbol, 0)
            
            if 'short_holdings' in record and record['short_holdings']:
                for symbol in all_symbols:
                    row[f'short_holding_{symbol}'] = record['short_holdings'].get(symbol, 0)
            
            if 'short_shares' in record and record['short_shares']:
                for symbol in all_symbols:
                    row[f'short_shares_{symbol}'] = record['short_shares'].get(symbol, 0)
                    
            holdings_data.append(row)
        
        df_holdings = pd.DataFrame(holdings_data)
        
        holdings_file_path = file_path.replace('.csv', '_holdings.csv')
        df_holdings.to_csv(holdings_file_path, index=False)
        print(f"Holdings and short positions saved to: {holdings_file_path}")
        
        return file_path

    def _calculate_portfolio_weights(self, state):
        """Calculate portfolio weights for the current state."""
        if self.config['portfolio'].get('use_mean_var', False):
            return self._mean_variance_optimization()
        elif self.config['portfolio'].get('use_momentum', False):
            return self._momentum_strategy()
        elif self.config['portfolio'].get('equal_weight', False):
            return self._equal_weight_strategy()
        else:
            return self.agent.predict(state)
            
    def _mean_variance_optimization(self):
        """Mean-variance optimization strategy."""
        window = 60
        returns = pd.DataFrame()
        
        for symbol in self.current_stocks:
            prices = self.data[symbol]['close'][-window:]
            returns[symbol] = prices.pct_change().dropna()
        
        mu = returns.mean()
        sigma = returns.cov()
        
        try:
            from scipy.optimize import minimize
            
            def objective(weights):
                portfolio_return = np.sum(weights * mu)
                portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(sigma, weights)))
                return -(portfolio_return - 0.02) / portfolio_risk
            
            n_assets = len(self.current_stocks)
            constraints = (
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            )
            bounds = tuple((0, 1) for _ in range(n_assets))
            
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
        """Momentum strategy using 20-day price momentum."""
        momentum_scores = {}
        window = 20
        
        for symbol in self.current_stocks:
            try:
                prices = self.data[symbol]['close'][-window:]
                momentum = (prices.iloc[-1] / prices.iloc[0] - 1)
                momentum_scores[symbol] = momentum
            except Exception as e:
                print(f"Error computing momentum for {symbol}: {e}")
                momentum_scores[symbol] = 0
        
        total_score = sum(max(0, score) for score in momentum_scores.values())
        if total_score > 0:
            weights = {symbol: max(0, score) / total_score 
                      for symbol, score in momentum_scores.items()}
        else:
            weights = self._equal_weight_strategy()
        
        return weights
    
    def _equal_weight_strategy(self):
        """Equal-weight strategy."""
        equal_weight = 1.0 / len(self.current_stocks) if self.current_stocks else 0
        return {symbol: equal_weight for symbol in self.current_stocks} 

    def _ensure_valid_state(self):
        """Ensure environment state is valid."""
        if self.cash < 0:
            print(f"Warning: negative cash detected ({self.cash:.2f}), resetting to 0")
            self.cash = 0
            
            holdings_value = sum(self.holdings.values()) if self.holdings else 0
            self.portfolio_value = holdings_value + self.cash
            
            total_value = self.portfolio_value
            if total_value > 0:
                self.current_weights = {symbol: value/total_value for symbol, value in self.holdings.items()}
            else:
                self.current_weights = {}
        
        if self.portfolio_value < 0:
            print(f"Warning: negative portfolio value ({self.portfolio_value:.2f}), resetting to cash value")
            self.portfolio_value = self.cash

