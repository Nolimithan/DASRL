"""
Data loader factory module.

Select the appropriate data loader based on configuration.
"""

from .data_loader import DataLoader
from .nasdaq_data_loader import NasdaqDataLoader
from .local_csv_loader import LocalCSVDataLoader


def create_data_loader(config, proxy=None, enable_proxy=False):
    """
    Create a data loader instance.

    Args:
        config (dict): Config dictionary.
        proxy (str, optional): Proxy server address.
        enable_proxy (bool): Whether to enable proxy.

    Returns:
        DataLoader or NasdaqDataLoader: Loader instance created by config.
    """
    # Read data source config
    data_source = config['data'].get('data_source', 'china')
    
    # Read common loader parameters
    start_date = config['data']['start_date']
    end_date = config['data']['end_date']
    features = config['features']['technical_indicators']
    symbols = config['data']['symbols']
    data_dir = config['data'].get('data_dir', 'data/stock_data')
    offline_mode = config['data'].get('offline', False)
    lookback_days = config['lim'].get('lookback_days', 90)
    
    if data_source.lower() in ('nasdaq', 'finance'):
        # Use Nasdaq loader (also for finance ETF data)
        print(f"Using {data_source} data loader (based on yfinance)")
        return NasdaqDataLoader(
            start_date=start_date,
            end_date=end_date,
            symbols=symbols,
            features=features,
            data_dir=data_dir,
            lookback_days=lookback_days,
            proxy=proxy,
            enable_proxy=enable_proxy,
            offline_mode=offline_mode
        )
    elif data_source.lower() in ('local', 'csv'):
        print("Using local CSV data loader")
        return LocalCSVDataLoader(
            start_date=start_date,
            end_date=end_date,
            symbols=symbols,
            features=features,
            data_dir=data_dir,
            lookback_days=lookback_days,
            benchmark_hint=config['data'].get('benchmark')
        )
    else:
        # Use domestic data loader (default)
        print("Using domestic data loader")
        return DataLoader(
            start_date=start_date,
            end_date=end_date,
            symbols=symbols,
            features=features,
            data_dir=data_dir,
            lookback_days=lookback_days
        ) 

