"""
Configuration module

Load and manage configuration files.
"""

import os
import yaml
import json


def get_config(config_path='config/default.yaml'):
    """Load configuration file.

    Create a default config file if it does not exist.

    Args:
        config_path: Config file path.

    Returns:
        dict: Config dictionary.
    """
    if not os.path.exists(config_path):
        create_default_config_file(config_path)
        
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    return config


def _load_yaml(yaml_path):
    """Load a YAML config file.

    Args:
        yaml_path (str): YAML file path.

    Returns:
        dict: Config dictionary.
    """
    with open(yaml_path, 'r', encoding='utf-8') as f:
        try:
            config = yaml.safe_load(f)
            return config
        except yaml.YAMLError as e:
            raise ValueError(f"YAML parse error: {e}")


def _load_json(json_path):
    """Load a JSON config file.

    Args:
        json_path (str): JSON file path.

    Returns:
        dict: Config dictionary.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        try:
            config = json.load(f)
            return config
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON parse error: {e}")


def save_config(config, output_path):
    """Save configuration.

    Persist the config dictionary to a file.

    Args:
        config (dict): Config dictionary.
        output_path (str): Output file path.
    """
    # Create directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Choose serializer based on file extension
    _, ext = os.path.splitext(output_path)
    
    if ext.lower() in ['.yaml', '.yml']:
        _save_yaml(config, output_path)
    elif ext.lower() == '.json':
        _save_json(config, output_path)
    else:
        raise ValueError(f"Unsupported config file format: {ext}")


def _save_yaml(config, output_path):
    """Save config to a YAML file.

    Args:
        config (dict): Config dictionary.
        output_path (str): Output file path.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def _save_json(config, output_path):
    """Save config to a JSON file.

    Args:
        config (dict): Config dictionary.
        output_path (str): Output file path.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def create_default_config_file(config_path='config/default.yaml'):
    """Create the default configuration file.

    Args:
        config_path: Config file path.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    default_config = {
        'data': {
            # Use the 2024 date range
            'start_date': '2024-01-01',
            'end_date': '2024-12-31',
            'symbols': [
                # Representative A-share semiconductor stocks (30)
                '688981',  # SMIC
                '002371',  # NAURA Technology Group
                '603501',  # Will Semiconductor
                '002049',  # Unigroup Guoxin Microelectronics
                '603986',  # GigaDevice
                '688012',  # AMEC
                '300458',  # Allwinner Technology
                '300782',  # Maxscend Microelectronics
                '688037',  # Kingsemi
                '603290',  # StarPower Semiconductor
                '300223',  # Beijing Junzheng (Ingenic)
                '300661',  # SG Micro
                '600745',  # Wingtech Technology
                '688099',  # Amlogic
                '688008',  # Montage Technology
                '688126',  # Shanghai Silicon Industry
                '600584',  # JCET Group
                '688396',  # China Resources Microelectronics
                '603160',  # Goodix Technology
                '688521',  # VeriSilicon
                '688516',  # Autowell
                '688536',  # 3PEAK
                '688256',  # Cambricon
                '300604',  # Changchuan Technology
                '688187',  # CRRC Times Electric
                '300671',  # Fuman Microelectronics
                '688200',  # Huafeng Test & Control
                '688233',  # Shengong
                '688268',  # Huate Gas
                '603690',  # Zhichun Technology
            ],
            'benchmark': '000001'  # SSE Composite Index
        },
        
        'lim': {
            'window_size': 60,  # 60-day window for LIM
            'min_LIM_value': 0.1,  # Minimum LIM value
            'rebalance_threshold': 0.3  # Rebalance threshold
        },
        
        'portfolio': {
            'size': 3,  # Number of stocks in portfolio
            'initial_capital': 1000000,  # Initial capital
            'min_stock_count': 3,  # Minimum number of holdings
            'min_holding_days': 10  # Minimum holding days
        },
        
        'trading': {
            'commission_rate': 0.001,  # Commission rate
            'rebalance_period': 30,  # Rebalance period (days)
            'stop_loss_threshold': 0.15,  # Stop-loss threshold
            'stop_loss_increment': 0.03,  # Stop-loss increment
            'max_stop_loss': 0.21,  # Maximum stop-loss threshold
            'target_cash_ratio': 0.3  # Target cash ratio (after stop-loss)
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
                'lambda_1': 0.5,  # Portfolio variance penalty weight
                'lambda_2': 0.2,  # Turnover penalty weight
                'lambda_3': 1.0   # Drawdown penalty weight
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
        
    print(f"Default config file created: {config_path}")


if __name__ == '__main__':
    # If run as a script, create the default config file
    create_default_config_file() 

