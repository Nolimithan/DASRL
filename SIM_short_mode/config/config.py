"""
Configuration loading and saving utilities.
"""

import os
import yaml
import json


def get_config(config_path='config/default.yaml'):
    """Load configuration from YAML, creating default if missing."""
    if not os.path.exists(config_path):
        create_default_config_file(config_path)
        
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    return config


def _load_yaml(yaml_path):
    """Load a YAML configuration file."""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        try:
            config = yaml.safe_load(f)
            return config
        except yaml.YAMLError as e:
            raise ValueError(f"YAML parse error: {e}")


def _load_json(json_path):
    """Load a JSON configuration file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        try:
            config = json.load(f)
            return config
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON parse error: {e}")


def save_config(config, output_path):
    """Save configuration to a file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    _, ext = os.path.splitext(output_path)
    
    if ext.lower() in ['.yaml', '.yml']:
        _save_yaml(config, output_path)
    elif ext.lower() == '.json':
        _save_json(config, output_path)
    else:
        raise ValueError(f"Unsupported config format: {ext}")


def _save_yaml(config, output_path):
    """Save configuration to a YAML file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def _save_json(config, output_path):
    """Save configuration to a JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def create_default_config_file(config_path='config/default.yaml'):
    """Create a default configuration file."""
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    default_config = {
        'data': {
            'data_source': 'nasdaq',
            'start_date': '2019-06-12',
            'end_date': '2024-04-05',
            'data_dir': 'data/nasdaq_etf',
            'symbols': [
                'AVGO',
                'NVDA',
                'TXN',
                'QCOM',
                'AMD',
                'KLAC',
                'AMAT',
                'LRCX',
                'INTC',
                'MPWR',
                'NXPI',
                'ASML',
                'ADI',
                'MU',
                'TSM',
                'MCHP',
                'MRVL',
                'ON',
                'TER',
                'QRVO',
                'SWKS',
                'LSCC',
                'SMTC',
                'WOLF',
                'STM',
                'OLED',
                'NTNX',
                'DIOD',
                'POWI',
            ],
            'benchmark': 'SOXX'
        },
        
        'lim': {
            'window_size': 60,
            'min_LIM_value': 0.1,
            'rebalance_threshold': 0.3
        },
        
        'portfolio': {
            'size': 3,
            'initial_capital': 1000000,
            'min_stock_count': 3,
            'min_holding_days': 10
        },
        
        'trading': {
            'commission_rate': 0.001,
            'rebalance_period': 30,
            'stop_loss_threshold': 0.15,
            'stop_loss_increment': 0.03,
            'max_stop_loss': 0.21,
            'target_cash_ratio': 0.3
        },
        
        'rl': {
            'learning_rate': 3e-4,
            'gamma': 0.99,
            'clip_ratio': 0.2,
            'value_coef': 0.5,
            'entropy_coef': 0.01,
            'epochs': 50,
            'steps_per_epoch': 2000,
            'reward_params': {
                'lambda_1': 0.5,
                'lambda_2': 0.2,
                'lambda_3': 1.0
            }
        },
        
        'features': {
            'technical_indicators': [
                'SMA20', 'SMA60', 'EMA12', 'EMA26', 'MACD', 
                'RSI14', 'ATR', 'BOLL_UPPER', 'BOLL_LOWER', 'VOL20'
            ],
            'fundamental_features': [
                'PE', 'PB', 'ROE', 'NET_PROFIT_YOY'
            ]
        },
        
        'paths': {
            'model_save_path': 'models/ppo_model',
            'model_load_path': 'models/ppo_model',
            'data_path': 'data/',
            'results_path': 'results/',
            'visualization_path': 'visualization/'
        },
        
        'environment': {
            'portfolio_size': 3,
            'initial_capital': 1000000,
            'commission_rate': 0.001,
            'stop_loss_threshold': 0.15,
            'rebalance_period': 30
        },
        
        'ablation': {
            'run_equal_weight': True,
            'run_momentum': True,
            'run_mean_var': True,
            'run_no_LIM': True,
            'run_no_exit_rule': True,
            'run_no_cov_penalty': True,
            'run_fixed_30day': True,
            'run_literature_rl': True
        }
    }
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(default_config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        
    print(f"Default config created: {config_path}")


if __name__ == '__main__':
    create_default_config_file() 

