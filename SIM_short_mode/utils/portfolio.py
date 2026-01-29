"""
Portfolio management and analysis utilities.
"""

import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt


class Portfolio:
    """Portfolio manager for running strategies and tracking history."""
    
    def __init__(self, initial_capital=1000000, commission_rate=0.003):
        """Initialize portfolio."""
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.portfolio_value = initial_capital
        self.max_portfolio_value = initial_capital
        self.current_holdings = {}
        
        self.history = {
            'date': [],
            'portfolio_value': [],
            'daily_return': [],
            'holdings': [],
            'transactions': [],
            'transaction_costs': []
        }
        
        self.performance_metrics = {}
    
    def run_strategy(self, agent, env, start_date=None, end_date=None):
        """Run the strategy using an agent in the environment."""
        state = env.reset(start_date=start_date, end_date=end_date)
        
        done = False
        total_reward = 0
        
        self._record_state(
            date=env.market_data.iloc[0]['date'] if env.market_data is not None else datetime.now(),
            portfolio_value=env.portfolio_value,
            holdings=env.current_stocks,
            weights=list(env.current_weights.values()) if isinstance(env.current_weights, dict) else env.current_weights.tolist(),
            transactions={},
            transaction_cost=0
        )
        
        while not done:
            action = agent.predict(state)
            
            next_state, reward, done, info = env.step(action)
            
            self._record_state(
                date=env.market_data.iloc[env.current_step]['date'] if env.market_data is not None else datetime.now(),
                portfolio_value=info['portfolio_value'],
                holdings=info['current_stocks'],
                weights=list(info['weights'].values()) if isinstance(info['weights'], dict) else info['weights'].tolist(),
                transactions=env.today_trades,
                transaction_cost=info['transaction_cost']
            )
            
            state = next_state
            total_reward += reward
        
        self._calculate_performance_metrics()
        
        return self.performance_metrics
    
    def run_ablation_studies(self, ablation_config, env):
        """Run ablation studies for multiple strategies."""
        results = {}
        
        baseline_agent = ablation_config.get('baseline_agent')
        if baseline_agent:
            print("Running baseline strategy...")
            baseline_results = self.run_strategy(baseline_agent, env)
            results['baseline'] = baseline_results
        
        no_LIM_agent = ablation_config.get('no_LIM_agent')
        no_LIM_env = ablation_config.get('no_LIM_env')
        if no_LIM_agent and no_LIM_env:
            print("Running no-LIM* strategy...")
            no_LIM_portfolio = Portfolio(
                initial_capital=self.initial_capital,
                commission_rate=self.commission_rate
            )
            no_LIM_results = no_LIM_portfolio.run_strategy(no_LIM_agent, no_LIM_env)
            results['no_LIM'] = no_LIM_results
        
        fixed_30day_agent = ablation_config.get('fixed_30day_agent')
        fixed_30day_env = ablation_config.get('fixed_30day_env')
        if fixed_30day_agent and fixed_30day_env:
            print("Running fixed 30-day rebalance strategy...")
            fixed_30day_portfolio = Portfolio(
                initial_capital=self.initial_capital,
                commission_rate=self.commission_rate
            )
            fixed_30day_results = fixed_30day_portfolio.run_strategy(fixed_30day_agent, fixed_30day_env)
            results['fixed_30day'] = fixed_30day_results
        
        no_cov_penalty_agent = ablation_config.get('no_cov_penalty_agent')
        no_cov_penalty_env = ablation_config.get('no_cov_penalty_env')
        if no_cov_penalty_agent and no_cov_penalty_env:
            print("Running no-covariance-penalty strategy...")
            no_cov_penalty_portfolio = Portfolio(
                initial_capital=self.initial_capital,
                commission_rate=self.commission_rate
            )
            no_cov_penalty_results = no_cov_penalty_portfolio.run_strategy(no_cov_penalty_agent, no_cov_penalty_env)
            results['no_cov_penalty'] = no_cov_penalty_results
        
        no_exit_rule_agent = ablation_config.get('no_exit_rule_agent')
        no_exit_rule_env = ablation_config.get('no_exit_rule_env')
        if no_exit_rule_agent and no_exit_rule_env:
            print("Running no-stop-loss strategy...")
            no_exit_rule_portfolio = Portfolio(
                initial_capital=self.initial_capital,
                commission_rate=self.commission_rate
            )
            no_exit_rule_results = no_exit_rule_portfolio.run_strategy(no_exit_rule_agent, no_exit_rule_env)
            results['no_exit_rule'] = no_exit_rule_results
        
        equal_weight_results = ablation_config.get('equal_weight_results')
        if equal_weight_results:
            results['equal_weight'] = equal_weight_results
        
        mean_var_results = ablation_config.get('mean_var_results')
        if mean_var_results:
            results['mean_var'] = mean_var_results
        
        momentum_results = ablation_config.get('momentum_results')
        if momentum_results:
            results['momentum'] = momentum_results
        
        literature_rl_results = ablation_config.get('literature_rl_results')
        if literature_rl_results:
            results['literature_rl'] = literature_rl_results
        
        return results
    
    def _record_state(self, date, portfolio_value, holdings, weights, transactions, transaction_cost):
        """Record current portfolio state."""
        self.portfolio_value = portfolio_value
        
        self.max_portfolio_value = max(self.max_portfolio_value, portfolio_value)
        
        self.current_holdings = {symbol: weight for symbol, weight in zip(holdings, weights)}
        
        if len(self.history['portfolio_value']) > 0:
            daily_return = (portfolio_value / self.history['portfolio_value'][-1]) - 1
        else:
            daily_return = 0
        
        self.history['date'].append(date)
        self.history['portfolio_value'].append(portfolio_value)
        self.history['daily_return'].append(daily_return)
        self.history['holdings'].append(holdings.copy() if isinstance(holdings, list) else list(holdings))
        self.history['transactions'].append(transactions.copy())
        self.history['transaction_costs'].append(transaction_cost)
    
    def _calculate_performance_metrics(self):
        """Compute portfolio performance metrics."""
        df = pd.DataFrame({
            'date': self.history['date'],
            'portfolio_value': self.history['portfolio_value'],
            'daily_return': self.history['daily_return']
        })
        
        if len(df) > 1:
            total_return = (df['portfolio_value'].iloc[-1] / df['portfolio_value'].iloc[0]) - 1
            years = len(df) / 252
            annual_return = (1 + total_return) ** (1 / years) - 1
        else:
            total_return = 0
            annual_return = 0
        
        if len(df) > 1:
            volatility = df['daily_return'].std() * np.sqrt(252)
        else:
            volatility = 0
        
        if volatility > 0:
            sharpe_ratio = annual_return / volatility
        else:
            sharpe_ratio = 0
        
        negative_returns = df[df['daily_return'] < 0]['daily_return']
        if len(negative_returns) > 0:
            downside_deviation = negative_returns.std() * np.sqrt(252)
            sortino_ratio = annual_return / downside_deviation if downside_deviation > 0 else 0
        else:
            sortino_ratio = 0
        
        running_max = df['portfolio_value'].cummax()
        drawdown = (running_max - df['portfolio_value']) / running_max
        max_drawdown = drawdown.max()
        
        win_days = len(df[df['daily_return'] > 0])
        win_rate = win_days / len(df) if len(df) > 0 else 0
        
        turnover = sum(self.history['transaction_costs']) / (2 * self.commission_rate * self.initial_capital)
        
        value_changes = np.diff(df['portfolio_value'])
        significant_drops = (value_changes / df['portfolio_value'].iloc[:-1]) < -0.1
        stop_loss_count = significant_drops.sum()
        
        cost_percentage = sum(self.history['transaction_costs']) / self.initial_capital
        
        self.performance_metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'turnover': turnover,
            'stop_loss_count': stop_loss_count,
            'cost_percentage': cost_percentage
        }
        
        return self.performance_metrics
    
    def get_history_dataframe(self):
        """Return history as a DataFrame."""
        return pd.DataFrame({
            'date': self.history['date'],
            'portfolio_value': self.history['portfolio_value'],
            'daily_return': self.history['daily_return']
        })
    
    def get_performance_summary(self):
        """Return performance metrics summary."""
        if not self.performance_metrics:
            self._calculate_performance_metrics()
        
        return self.performance_metrics 

