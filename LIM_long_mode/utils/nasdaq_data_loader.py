"""
Nasdaq semiconductor ETF data loader (simplified).

Reads and preprocesses Nasdaq semiconductor ETF and constituents, and computes indicators.
Uses yfinance only for simplicity and efficiency.
"""

import os
import pandas as pd
import numpy as np
import talib as ta
import yfinance as yf
try:
    import yaml
except Exception:
    yaml = None
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from datetime import datetime, timedelta
import time
import random
from typing import Dict, List, Optional
import warnings

warnings.filterwarnings("ignore")

# Proxy configuration helper
def setup_proxy(proxy_url=None, enable_proxy=False):
    """Configure proxy settings."""
    if enable_proxy and proxy_url:
        print(f"Enabling yfinance proxy: {proxy_url}")
        os.environ['HTTP_PROXY'] = proxy_url
        os.environ['HTTPS_PROXY'] = proxy_url
        os.environ['ENABLE_YFINANCE_PROXY'] = 'true'
        
        try:
            import requests
            session = requests.Session()
            session.proxies = {
                'http': proxy_url,
                'https': proxy_url
            }
            # Set global session for yfinance
            yf.download._session = session
            print("✅ yfinance proxy configured")
            return session
        except Exception as e:
            print(f"Warning while setting yfinance proxy: {e}")
            return None
    else:
        print("yfinance proxy not enabled")
        return None


class NasdaqDataLoader:
    """
    Nasdaq data loader (simplified, yfinance only).
    Efficient and stable data retrieval.
    """
    
    def __init__(self, start_date: str, end_date: str, symbols: List[str],
                 features: List[str], data_dir: str = 'data/nasdaq',
                 lookback_days: int = 90, proxy: str = None, enable_proxy: bool = False,
                 offline_mode: bool = False):
        """
        Initialize the Nasdaq data loader.
        
        Args:
            start_date: Start date.
            end_date: End date.
            symbols: List of tickers.
            features: Technical indicator features.
            data_dir: Data storage directory.
            lookback_days: Extra lookback days.
            proxy: Proxy server address, e.g. 'http://127.0.0.1:7890'.
            enable_proxy: Whether to enable proxy.
        """
        self.start_date = start_date
        self.end_date = end_date
        self.symbols = symbols
        self.features = features
        self.data_dir = data_dir
        self.lookback_days = lookback_days
        self.proxy = proxy
        self.enable_proxy = enable_proxy
        self.offline_mode = offline_mode
        
        # Adjust start date to include enough history
        adjusted_start = pd.to_datetime(start_date) - pd.Timedelta(days=lookback_days)
        self.adjusted_start_date = adjusted_start.strftime('%Y-%m-%d')
        self.data_start_date = self.adjusted_start_date
        
        print(f"Adjusted data start date: {self.data_start_date} (lookback {lookback_days} days)")
        
        # Configure proxy
        if enable_proxy and proxy:
            self.session = setup_proxy(proxy, enable_proxy)
        
        # Data storage
        self.all_stock_data = {}
        self.benchmark = None
        self.data = None
        self.processed_data = None
        self.scaler = StandardScaler()
        
        # Ensure data directory exists
        os.makedirs(data_dir, exist_ok=True)
    
    def _get_stock_data_from_yfinance(self, symbol):
        """Fetch stock data from yfinance with caching."""
        # Build cache file path
        cache_file = os.path.join(self.data_dir, f"{symbol}_{self.data_start_date}_{self.end_date}.csv")
        
        # Check cache
        if os.path.exists(cache_file):
            try:
                df = pd.read_csv(cache_file)
                print(f"Loading {symbol} data from cache")
                
                # Ensure date column is datetime64[ns]
                df['date'] = pd.to_datetime(df['date'])
                return df
            except Exception as e:
                print(f"Failed to read cache: {e}, refetching")

        # Offline mode: require local cache to avoid online requests
        if self.offline_mode:
            raise RuntimeError(f"Offline mode enabled; cache not found: {cache_file}")
        
        try:
            # Fetch data from API
            print(f"Fetching {symbol} data, range: {self.data_start_date} to {self.end_date}")
            
            # Prepare proxy settings
            session = None
            if self.enable_proxy and self.proxy:
                print(f"Using proxy for {symbol} data: {self.proxy}")
                try:
                    import requests
                    session = requests.Session()
                    session.proxies = {
                        'http': self.proxy,
                        'https': self.proxy
                    }
                    # Configure timeouts and retries
                    session.timeout = 30
                    from requests.adapters import HTTPAdapter
                    from urllib3.util.retry import Retry
                    retry_strategy = Retry(
                        total=3,
                        backoff_factor=1,
                        status_forcelist=[429, 500, 502, 503, 504],
                    )
                    adapter = HTTPAdapter(max_retries=retry_strategy)
                    session.mount("http://", adapter)
                    session.mount("https://", adapter)
                except Exception as e:
                    print(f"Failed to set proxy session: {e}")
                    session = None
            
            # Fetch via yfinance, passing session when available
            if session:
                ticker = yf.Ticker(symbol, session=session)
            else:
                ticker = yf.Ticker(symbol)
                
            df = ticker.history(start=self.data_start_date, end=self.end_date, interval="1d")
            
            if df is None or df.empty:
                print(f"Warning: failed to fetch data for {symbol}")
                return None
            
            # Reset index, move Date from index to column
            df = df.reset_index()
            
            # Normalize column names to lowercase
            df.columns = [col.lower() for col in df.columns]
            df.rename(columns={
                'date': 'date',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            }, inplace=True)
            
            # Ensure date column is datetime64[ns] and strip timezone
            df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
            
            # Add symbol columns
            df['symbol'] = symbol
            df['ts_code'] = symbol
            
            # Compute daily returns
            df['daily_return'] = df['close'].pct_change()
            
            # Save to cache
            df.to_csv(cache_file, index=False)
            
            print(f"Fetched {symbol} data, {len(df)} rows")
            return df
            
        except Exception as e:
            print(f"Error fetching {symbol} data: {e}")
            # If proxy-related error, provide hints
            if "proxy" in str(e).lower() or "connection" in str(e).lower():
                print(f"Proxy connection failed. Check proxy settings: {self.proxy}")
                print("Suggestions:")
                print("1. Ensure the proxy server is running")
                print("2. Verify proxy address and port")
                print("3. Test the proxy connection in a browser")
            return None
    
    def _get_benchmark_from_yfinance(self):
        """Fetch benchmark index data from yfinance."""
        # Prefer benchmark from config or env
        preferred = None
        # Environment variable first (optional)
        preferred = os.environ.get('YF_BENCHMARK', preferred)
        # YAML config (default LIMPPO_CNN/config/default.yaml)
        if preferred is None and yaml is not None:
            try:
                default_cfg_path = os.path.join('LIMPPO_CNN', 'config', 'default.yaml')
                if os.path.exists(default_cfg_path):
                    with open(default_cfg_path, 'r', encoding='utf-8') as f:
                        cfg = yaml.safe_load(f)
                        preferred = cfg.get('data', {}).get('benchmark')
            except Exception:
                preferred = None

        # If benchmark explicitly specified (e.g., ^GDAXI), only try that
        if preferred:
            df = self._get_stock_data_from_yfinance(preferred)
            if df is not None and not df.empty:
                print(f"Benchmark loaded: {preferred}, points: {len(df)}")
                return df
            print(f"❌ Failed to fetch configured benchmark: {preferred}")
            return None

        # Otherwise try common indices/ETFs in priority order (includes DAX)
        benchmark_symbols = ['^GDAXI', 'DAX', 'EXS1.DE', 'EWG', '^IXIC', 'QQQ', 'SOXX', 'SMH']
        for symbol in benchmark_symbols:
            df = self._get_stock_data_from_yfinance(symbol)
            if df is not None and not df.empty:
                print(f"Benchmark loaded: {symbol}, points: {len(df)}")
                return df
        
        print("Warning: failed to fetch benchmark, using synthetic benchmark")
        return None
    
    def load_data(self):
        """Load stock data, compute indicators, and build benchmark."""
        print(f"Loading data for {len(self.symbols)} symbols...")
        
        all_data = []
        success_count = 0
        
        for i, symbol in enumerate(self.symbols, 1):
            print(f"[{i}/{len(self.symbols)}] Fetching {symbol} data...")
            df = self._get_stock_data_from_yfinance(symbol)
            
            if df is not None and not df.empty:
                # Ensure data sorted by date (old to new)
                df = df.sort_values('date').reset_index(drop=True)
                
                # Compute technical indicators
                df = self._calculate_technical_indicators(df)
                
                # Append to list
                all_data.append(df)
                
                # Store in dict for environment
                self.all_stock_data[symbol] = df
                
                success_count += 1
                print(f"✅ {symbol} data loaded")
            else:
                print(f"❌ Skipping {symbol}: no data")
            
            # Add delay between requests to avoid API limits (5s)
            if i < len(self.symbols):
                print("Waiting 5 seconds to avoid API rate limits...")
                time.sleep(5)
        
        # Merge all data (compatibility with original interface)
        if all_data:
            self.data = pd.concat(all_data, ignore_index=True)
            if 'date' in self.data.columns:
                self.data['date'] = pd.to_datetime(self.data['date'])
            print(f"Loaded data for {success_count} symbols")
        else:
            # Strict mode: if no DAX constituents, terminate experiment
            raise RuntimeError("Failed to fetch any DAX constituents; aborting")
        
        # Load benchmark data
        self._load_benchmark_data()
        
        # Drop missing values
        if not self.data.empty:
            self.data = self.data.dropna().reset_index(drop=True)
        
        print(f"Data load complete: {success_count}/{len(self.symbols)} symbols")

    def _load_benchmark_data(self):
        """Load benchmark data."""
        benchmark_symbol = '^IXIC'  # Nasdaq Composite Index

        print(f"Loading benchmark index data: {benchmark_symbol}")
        self.benchmark = self._get_benchmark_from_yfinance()
        
        if self.benchmark is None or self.benchmark.empty:
            # If config explicitly specifies benchmark, fail strictly
            strict_symbol = None
            if yaml is not None:
                try:
                    default_cfg_path = os.path.join('LIMPPO_CNN', 'config', 'default.yaml')
                    if os.path.exists(default_cfg_path):
                        with open(default_cfg_path, 'r', encoding='utf-8') as f:
                            cfg = yaml.safe_load(f)
                            strict_symbol = cfg.get('data', {}).get('benchmark')
                except Exception:
                    strict_symbol = None

            if strict_symbol:
                raise RuntimeError(f"Benchmark {strict_symbol} fetch failed; aborting")

            print("Building synthetic benchmark...")
            # Create synthetic benchmark from average returns
            if self.all_stock_data:
                # Collect per-stock date/return data
                all_returns = []
                
                for symbol, stock_data in self.all_stock_data.items():
                    if not stock_data.empty and 'daily_return' in stock_data.columns:
                        stock_returns = stock_data[['date', 'daily_return']].copy()
                        stock_returns['symbol'] = symbol
                        all_returns.append(stock_returns)
                
                if all_returns:
                    # Merge all return data
                    combined_returns = pd.concat(all_returns, ignore_index=True)
                    
                    # Group by date and compute average daily return
                    daily_avg_returns = combined_returns.groupby('date')['daily_return'].mean().reset_index()
                    daily_avg_returns = daily_avg_returns.sort_values('date').reset_index(drop=True)
                    
                    # Build benchmark DataFrame
                    self.benchmark = daily_avg_returns.copy()
                    self.benchmark['close'] = 1000  # Initial value
                    self.benchmark['open'] = 1000   
                    self.benchmark['high'] = 1000
                    self.benchmark['low'] = 1000
                    self.benchmark['volume'] = 0
                    
                    # Compute cumulative returns
                    for i in range(1, len(self.benchmark)):
                        prev_close = self.benchmark.loc[i-1, 'close']
                        daily_return = self.benchmark.loc[i, 'daily_return']
                        close_price = prev_close * (1 + daily_return) if not pd.isna(daily_return) else prev_close
                        
                        self.benchmark.loc[i, 'close'] = close_price
                        self.benchmark.loc[i, 'open'] = close_price
                        self.benchmark.loc[i, 'high'] = close_price
                        self.benchmark.loc[i, 'low'] = close_price
                    
                    # Ensure date column is datetime64[ns]
                    self.benchmark['date'] = pd.to_datetime(self.benchmark['date'])
                    print("✅ Synthetic benchmark created")
                else:
                    print("❌ Error: cannot build benchmark, return data empty")
                    self.benchmark = pd.DataFrame()
            else:
                print("❌ Error: cannot build benchmark, stock data empty")
                self.benchmark = pd.DataFrame()
    
    def _calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute technical indicators."""
        try:
            # Ensure basic price columns exist
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in data.columns]
            
            if missing_cols:
                print(f"Warning: missing columns {missing_cols}, skipping related indicators")
            
            # Compute basic returns
            if 'close' in data.columns:
                data['daily_return'] = data['close'].pct_change()
                data['volatility'] = data['daily_return'].rolling(20).std()
                data['momentum'] = data['close'].pct_change(10)
            
            # Compute RSI
            if 'close' in data.columns:
                delta = data['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                data['rsi_14'] = 100 - (100 / (1 + rs))
            
            # Compute MACD
            if 'close' in data.columns:
                exp1 = data['close'].ewm(span=12).mean()
                exp2 = data['close'].ewm(span=26).mean()
                data['macd'] = exp1 - exp2
            
            # Compute Bollinger band width
            if 'close' in data.columns:
                rolling_mean = data['close'].rolling(20).mean()
                rolling_std = data['close'].rolling(20).std()
                data['bb_width'] = (rolling_std * 2) / rolling_mean
            
            # Compute ATR
            if all(col in data.columns for col in ['high', 'low', 'close']):
                high_low = data['high'] - data['low']
                high_close = np.abs(data['high'] - data['close'].shift())
                low_close = np.abs(data['low'] - data['close'].shift())
                tr = np.maximum(high_low, np.maximum(high_close, low_close))
                data['atr_14'] = tr.rolling(14).mean()
            
            # Compute relative strength
            if 'close' in data.columns:
                data['relative_strength'] = data['close'] / data['close'].rolling(50).mean()
            
            # Compute volume change
            if 'volume' in data.columns:
                data['volume_change'] = data['volume'].pct_change()
            
            # Fill NaN values
            data = data.ffill().bfill()
            
            return data
            
        except Exception as e:
            print(f"Error computing technical indicators: {e}")
            return data
    
    def calculate_technical_indicators(self, df):
        """Compute technical indicators (compatibility wrapper)."""
        return self._calculate_technical_indicators(df)
    
    def normalize_features(self, data=None):
        """Normalize features."""
        if data is None:
            data = self.data.copy()
        
        if data.empty:
            print("Error: no data available for normalization")
            return data
        
        # Get feature columns for scaling
        feature_cols = [col for col in data.columns if col not in ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']]
        
        # Scale features
        self.scaler.fit(data[feature_cols])
        data_scaled = data.copy()
        data_scaled[feature_cols] = self.scaler.transform(data[feature_cols])
        
        self.processed_data = data_scaled
        return data_scaled
    
    def get_data_for_date_range(self, start_date=None, end_date=None, symbols=None, normalized=True, use_pca=False, n_components=20):
        """Get data within a date range."""
        if self.data is None or self.data.empty:
            print("Error: no data loaded")
            return pd.DataFrame()
        
        # Convert to datetime64
        start = pd.to_datetime(start_date) if start_date else pd.to_datetime(self.start_date)
        end = pd.to_datetime(end_date) if end_date else pd.to_datetime(self.end_date)
        
        # Ensure date column is datetime64
        self.data['date'] = pd.to_datetime(self.data['date'])
        
        # Set symbol list
        syms = symbols if symbols else self.symbols
        
        # Filter data
        mask = (self.data['date'] >= start) & (self.data['date'] <= end) & (self.data['symbol'].isin(syms))
        filtered_data = self.data[mask].copy()
        
        if filtered_data.empty:
            print(f"Warning: no data for range {start} to {end}, symbols: {syms}")
            return pd.DataFrame()
        
        # Normalize if requested
        if normalized:
            filtered_data = self.normalize_features(filtered_data)
        
        return filtered_data
    
    def set_date_range(self, start_date, end_date):
        """Set date range for the data loader."""
        self.start_date = start_date
        self.end_date = end_date
        
        # If data already loaded, filter by new date range
        if self.data is not None and not self.data.empty:
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            
            # Filter data
            self.data['date'] = pd.to_datetime(self.data['date'])
            mask = (self.data['date'] >= start) & (self.data['date'] <= end)
            self.data = self.data[mask].reset_index(drop=True)
            
            # Filter benchmark data
            if self.benchmark is not None and not self.benchmark.empty:
                self.benchmark['date'] = pd.to_datetime(self.benchmark['date'])
                mask = (self.benchmark['date'] >= start) & (self.benchmark['date'] <= end)
                self.benchmark = self.benchmark[mask].reset_index(drop=True)
            
            # Filter all per-symbol data
            for symbol in self.all_stock_data:
                stock_data = self.all_stock_data[symbol]
                stock_data['date'] = pd.to_datetime(stock_data['date'])
                mask = (stock_data['date'] >= start) & (stock_data['date'] <= end)
                self.all_stock_data[symbol] = stock_data[mask].reset_index(drop=True)
                
            print(f"Date range updated to {start_date} - {end_date}, data filtered")
        else:
            print(f"Date range updated to {start_date} - {end_date}, will apply on next load")

