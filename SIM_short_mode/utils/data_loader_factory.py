"""
Data loader factory for selecting the appropriate loader.
"""

from .data_loader import DataLoader
from .nasdaq_data_loader import NasdaqDataLoader


def create_data_loader(config, proxy=None, enable_proxy=False):
    """Create a data loader instance from configuration."""
    data_source = config['data'].get('data_source', 'china')
    
    start_date = config['data']['start_date']
    end_date = config['data']['end_date']
    features = config['features']['technical_indicators']
    symbols = config['data']['symbols']
    data_dir = config['data'].get('data_dir', 'data/stock_data')
    lookback_days = config['sim'].get('lookback_days', 90)
    
    if (data_source.lower() == 'nasdaq' or 
        data_source.lower() == 'finance' or 
        data_source.lower() == 'hedge'):
        
        if data_source.lower() == 'hedge':
            print("Using hedge US-stock loader (yfinance) for short strategy")
            print(f"Stock universe size: {len(symbols)} US semiconductor downstream firms")
            data_dir = config['data'].get('data_dir', 'data/hedge_stock_data')
            
            benchmark = config['data'].get('benchmark', '^IXIC')
            print(f"Benchmark: {benchmark}")
        else:
            print(f"Using {data_source} loader (yfinance) for short strategy")
        
        if enable_proxy and proxy:
            print(f"Proxy enabled: {proxy}")
        else:
            print("Proxy disabled")
            
        return NasdaqDataLoader(
            start_date=start_date,
            end_date=end_date,
            symbols=symbols,
            features=features,
            data_dir=data_dir,
            lookback_days=lookback_days,
            proxy=proxy,
            enable_proxy=enable_proxy
        )
    else:
        print("Using domestic data loader for short strategy")
        return DataLoader(
            start_date=start_date,
            end_date=end_date,
            symbols=symbols,
            features=features,
            data_dir=data_dir,
            lookback_days=lookback_days
        ) 

