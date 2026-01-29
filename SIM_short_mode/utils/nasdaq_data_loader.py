"""
Nasdaq semiconductor ETF data loader.
"""

import os
import pandas as pd
import numpy as np
import talib as ta
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd


class NasdaqDataLoader:
    """Loader for Nasdaq ETF and constituent data."""
    
    def __init__(self, start_date, end_date, symbols, features=None, data_dir='data/nasdaq_etf', lookback_days=90, proxy=None, enable_proxy=False):
        """Initialize the data loader."""
        self.start_date = start_date
        self.end_date = end_date
        self.symbols = symbols
        self.data_dir = data_dir
        self.lookback_days = lookback_days
        self.proxy = proxy
        self.enable_proxy = enable_proxy
        
        if enable_proxy and proxy:
            self._setup_proxy()
        
        self.data_start_date = self._get_adjusted_start_date(start_date, lookback_days)
        print(f"Adjusted data start date: {self.data_start_date} (lookback {lookback_days} days)")
        
        self.default_features = [
            "RSI(14)", "MACD(12,26)", "BollingerBands(20)", 
            "ATR(14)", "Stochastic(14,3)", "ADX(14)", 
            "OBV", "CCI(20)", "VWAP", "Ichimoku"
        ]
        
        self.features = features if features else self.default_features
        
        self.data = None
        self.processed_data = None
        self.benchmark = None
        self.scaler = StandardScaler()
        
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.all_stock_data = {}
    
    def _setup_proxy(self):
        """Configure HTTP proxy settings."""
        try:
            import os
            import requests
            
            os.environ['HTTP_PROXY'] = self.proxy
            os.environ['HTTPS_PROXY'] = self.proxy
            
            print(f"Testing proxy connection: {self.proxy}")
            test_session = requests.Session()
            test_session.proxies = {
                'http': self.proxy,
                'https': self.proxy
            }
            
            test_response = test_session.get('https://finance.yahoo.com', timeout=10)
            if test_response.status_code == 200:
                print("Proxy connection test succeeded")
            else:
                print(f"Proxy connection test failed, status: {test_response.status_code}")
                
        except Exception as e:
            print(f"Proxy setup failed: {e}")
            print("Continuing without proxy...")
            self.enable_proxy = False
    
    def _get_adjusted_start_date(self, start_date, lookback_days):
        """Compute adjusted start date with lookback."""
        try:
            date_obj = pd.to_datetime(start_date)
            adjusted_date = date_obj - pd.Timedelta(days=lookback_days)
            return adjusted_date.strftime('%Y-%m-%d')
        except Exception as e:
            print(f"Date adjustment error: {e}. Using original date.")
            return start_date
    
    def _find_best_cache_file(self, symbol):
        """Find the best cache file covering the date range."""
        import glob
        import re
        
        exact_cache = os.path.join(self.data_dir, f"{symbol}_{self.data_start_date}_{self.end_date}.csv")
        if os.path.exists(exact_cache):
            return exact_cache
        
        pattern = os.path.join(self.data_dir, f"{symbol}_*.csv")
        candidates = glob.glob(pattern)
        
        if not candidates:
            return None
        
        required_start = pd.to_datetime(self.data_start_date)
        required_end = pd.to_datetime(self.end_date)
        
        best_file = None
        best_coverage = None
        
        for cache_file in candidates:
            filename = os.path.basename(cache_file)
            match = re.match(rf"{re.escape(symbol)}_(\d{{4}}-\d{{2}}-\d{{2}})_(\d{{4}}-\d{{2}}-\d{{2}})\.csv", filename)
            if not match:
                continue
            
            try:
                cache_start = pd.to_datetime(match.group(1))
                cache_end = pd.to_datetime(match.group(2))
                
                if cache_start <= required_start and cache_end >= required_end:
                    if best_coverage is None or (cache_end - cache_start) < (best_coverage[1] - best_coverage[0]):
                        best_file = cache_file
                        best_coverage = (cache_start, cache_end)
            except Exception:
                continue
        
        return best_file
    
    def _get_stock_data_from_yfinance(self, symbol):
        """Fetch stock data from yfinance or cache."""
        cache_file = self._find_best_cache_file(symbol)
        
        if cache_file is not None:
            df = pd.read_csv(cache_file)
            print(f"Loaded {symbol} from cache: {os.path.basename(cache_file)}")
            
            df['date'] = pd.to_datetime(df['date'])
            
            required_start = pd.to_datetime(self.data_start_date)
            required_end = pd.to_datetime(self.end_date)
            df = df[(df['date'] >= required_start) & (df['date'] <= required_end)]
            
            if not df.empty:
                return df
            else:
                print(f"Warning: cache file {cache_file} lacks required date range")
        
        new_cache_file = os.path.join(self.data_dir, f"{symbol}_{self.data_start_date}_{self.end_date}.csv")
        
        try:
            print(f"Fetching {symbol} data: {self.data_start_date} to {self.end_date}")
            
            if self.enable_proxy and self.proxy:
                import requests
                
                session = requests.Session()
                session.proxies.update({
                    'http': self.proxy,
                    'https': self.proxy
                })
                
                ticker = yf.Ticker(symbol, session=session)
            else:
                ticker = yf.Ticker(symbol)
            
            df = ticker.history(start=self.data_start_date, end=self.end_date, interval="1d")
            
            if df is None or df.empty:
                print(f"Warning: unable to fetch data for {symbol}")
                return None
            
            df = df.reset_index()
            
            df.columns = [col.lower() for col in df.columns]
            df.rename(columns={
                'date': 'date',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            }, inplace=True)
            
            df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
            
            df['symbol'] = symbol
            
            df['ts_code'] = symbol
            
            df['daily_return'] = df['close'].pct_change()
            
            df.to_csv(new_cache_file, index=False)
            
            print(f"Fetched {symbol} data: {len(df)} rows")
            return df
            
        except Exception as e:
            print(f"Error fetching {symbol} data: {e}")
            if self.enable_proxy and ("proxy" in str(e).lower() or "connection" in str(e).lower()):
                print(f"Proxy connection failed. Check proxy settings: {self.proxy}")
                print("Tips:")
                print("1. Ensure the proxy server is running")
                print("2. Verify the proxy host and port")
                print("3. Test the proxy connection in a browser")
            return None
    
    def _get_benchmark_from_yfinance(self):
        """Fetch benchmark index data from yfinance."""
        benchmark_symbols = ['SOXX', 'SMH', 'XSD', 'XLF']
        
        for symbol in benchmark_symbols:
            df = self._get_stock_data_from_yfinance(symbol)
            if df is not None and not df.empty:
                print(f"Benchmark fetched: {symbol}, points: {len(df)}")
                return df
        
        try:
            print("Falling back to Nasdaq index benchmark...")
            df = self._get_stock_data_from_yfinance('^IXIC')
            if df is not None and not df.empty:
                print(f"Nasdaq index benchmark fetched, points: {len(df)}")
                return df
        except Exception as e:
            print(f"Error fetching Nasdaq benchmark: {e}")
        
        print("Warning: unable to fetch benchmark data; using synthetic benchmark")
        return None
    
    def load_data(self):
        """Load stock data, compute indicators, and create benchmark."""
        print(f"Loading data for {len(self.symbols)} symbols...")
        
        all_data = []
        success_count = 0
        
        for symbol in self.symbols:
            print(f"Fetching data for {symbol}...")
            df = self._get_stock_data_from_yfinance(symbol)
            
            if df is not None and not df.empty:
                df = df.sort_values('date').reset_index(drop=True)
                
                df = self.calculate_technical_indicators(df)
                
                all_data.append(df)
                
                self.all_stock_data[symbol] = df
                
                success_count += 1
            else:
                print(f"Skipping {symbol}: no data")
        
        if all_data:
            self.data = pd.concat(all_data, ignore_index=True)
            if 'date' in self.data.columns:
                self.data['date'] = pd.to_datetime(self.data['date'])
            print(f"Loaded data for {success_count} symbols")
        else:
            print("Error: no stock data loaded")
            self.data = pd.DataFrame()
        
        self.load_benchmark()
        
        if not self.data.empty:
            self.data = self.data.dropna().reset_index(drop=True)
            self.set_date_range(self.start_date, self.end_date)
    
    def load_benchmark(self):
        """Load benchmark index data."""
        print("Fetching benchmark data...")
        self.benchmark = self._get_benchmark_from_yfinance()
        
        if self.benchmark is None or self.benchmark.empty:
            print("Creating synthetic benchmark...")
            if not self.data.empty:
                daily_avg_returns = self.data.groupby('date')['daily_return'].mean().reset_index()
                
                self.benchmark = daily_avg_returns.copy()
                self.benchmark['close'] = 1000
                self.benchmark['open'] = 1000
                self.benchmark['high'] = 1000
                self.benchmark['low'] = 1000
                self.benchmark['volume'] = 0
                
                for i in range(1, len(self.benchmark)):
                    close_price = self.benchmark.loc[i-1, 'close'] * (1 + self.benchmark.loc[i, 'daily_return'])
                    self.benchmark.loc[i, 'close'] = close_price
                    self.benchmark.loc[i, 'open'] = close_price
                    self.benchmark.loc[i, 'high'] = close_price
                    self.benchmark.loc[i, 'low'] = close_price
                
                self.benchmark['date'] = pd.to_datetime(self.benchmark['date'])
                print("Synthetic benchmark created")
            else:
                print("Error: cannot create benchmark, data is empty")
                self.benchmark = pd.DataFrame()
    
    def calculate_technical_indicators(self, df):
        """Calculate technical indicators for a DataFrame."""
        if len(df) < 30:
            print(f"Warning: insufficient data for {df['symbol'].iloc[0]}, skipping indicators")
            return df
        
        required_cols = ['open', 'high', 'low', 'close']
        for col in required_cols:
            if col not in df.columns:
                df[col] = df['close']
                print(f"Warning: missing {col} column, using close")
        
        if 'volume' not in df.columns:
            df['volume'] = 0
            print("Warning: missing volume column, using 0")
        
        for feature in self.features:
            try:
                if "RSI" in feature:
                    period = int(feature.split('(')[1].split(')')[0])
                    df[f'RSI_{period}'] = ta.RSI(df['close'].values, timeperiod=period)
                
                elif "MACD" in feature:
                    params = feature.split('(')[1].split(')')[0].split(',')
                    fast_period = int(params[0])
                    slow_period = int(params[1])
                    signal_period = 9
                    
                    macd, macdsignal, macdhist = ta.MACD(
                        df['close'].values, 
                        fastperiod=fast_period, 
                        slowperiod=slow_period, 
                        signalperiod=signal_period
                    )
                    df['MACD'] = macd
                    df['MACD_signal'] = macdsignal
                    df['MACD_hist'] = macdhist
                
                elif "BollingerBands" in feature:
                    period = int(feature.split('(')[1].split(')')[0])
                    upperband, middleband, lowerband = ta.BBANDS(
                        df['close'].values, 
                        timeperiod=period
                    )
                    df['BB_upper'] = upperband
                    df['BB_middle'] = middleband
                    df['BB_lower'] = lowerband
                    df['BB_width'] = (upperband - lowerband) / middleband
                
                elif "ATR" in feature:
                    period = int(feature.split('(')[1].split(')')[0])
                    df['ATR'] = ta.ATR(
                        df['high'].values, 
                        df['low'].values, 
                        df['close'].values, 
                        timeperiod=period
                    )
                
                elif "Stochastic" in feature:
                    params = feature.split('(')[1].split(')')[0].split(',')
                    k_period = int(params[0])
                    d_period = int(params[1])
                    
                    slowk, slowd = ta.STOCH(
                        df['high'].values, 
                        df['low'].values, 
                        df['close'].values, 
                        fastk_period=k_period,
                        slowk_period=3,
                        slowk_matype=0,
                        slowd_period=d_period,
                        slowd_matype=0
                    )
                    df['Stoch_K'] = slowk
                    df['Stoch_D'] = slowd
                
                elif "ADX" in feature:
                    period = int(feature.split('(')[1].split(')')[0])
                    df['ADX'] = ta.ADX(
                        df['high'].values, 
                        df['low'].values, 
                        df['close'].values, 
                        timeperiod=period
                    )
                
                elif "OBV" in feature:
                    if df['volume'].sum() == 0:
                        print(f"Warning: {df['symbol'].iloc[0]} volume is zero; skipping OBV")
                        df['OBV'] = 0
                    else:
                        df['OBV'] = ta.OBV(df['close'].values, df['volume'].values)
                
                elif "CCI" in feature:
                    period = int(feature.split('(')[1].split(')')[0])
                    df['CCI'] = ta.CCI(
                        df['high'].values, 
                        df['low'].values, 
                        df['close'].values, 
                        timeperiod=period
                    )
                
                elif "VWAP" in feature:
                    if df['volume'].sum() == 0:
                        print(f"Warning: {df['symbol'].iloc[0]} volume is zero; using simple average for VWAP")
                        df['VWAP'] = (df['high'] + df['low'] + df['close']) / 3
                    else:
                        df['VWAP'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
                
                elif "Ichimoku" in feature:
                    conversion_period = 9
                    base_period = 26
                    leading_span_b_period = 52
                    
                    high_9 = df['high'].rolling(window=conversion_period).max()
                    low_9 = df['low'].rolling(window=conversion_period).min()
                    df['Ichimoku_conv'] = (high_9 + low_9) / 2
                    
                    high_26 = df['high'].rolling(window=base_period).max()
                    low_26 = df['low'].rolling(window=base_period).min()
                    df['Ichimoku_base'] = (high_26 + low_26) / 2
                    
                    df['Ichimoku_spanA'] = ((df['Ichimoku_conv'] + df['Ichimoku_base']) / 2).shift(base_period)
                    
                    high_52 = df['high'].rolling(window=leading_span_b_period).max()
                    low_52 = df['low'].rolling(window=leading_span_b_period).min()
                    df['Ichimoku_spanB'] = ((high_52 + low_52) / 2).shift(base_period)
            
            except Exception as e:
                print(f"Error computing {feature}: {e}")
        
        df['return_30d'] = df['close'].pct_change(30)
        
        df['date'] = pd.to_datetime(df['date'])
        
        return df
    
    def get_data_for_date_range(self, start_date=None, end_date=None, symbols=None, normalized=True, use_pca=False, n_components=20):
        """Get data within a date range and symbol set."""
        if self.data is None or self.data.empty:
            print("Error: no data loaded")
            return pd.DataFrame()
        
        start = pd.to_datetime(start_date) if start_date else pd.to_datetime(self.start_date)
        end = pd.to_datetime(end_date) if end_date else pd.to_datetime(self.end_date)
        
        self.data['date'] = pd.to_datetime(self.data['date'])
        
        syms = symbols if symbols else self.symbols
        
        mask = (self.data['date'] >= start) & (self.data['date'] <= end) & (self.data['symbol'].isin(syms))
        filtered_data = self.data[mask].copy()
        
        if filtered_data.empty:
            print(f"Warning: no data for range {start} to {end}, symbols: {syms}")
            return pd.DataFrame()
        
        if normalized:
            filtered_data = self.normalize_features(filtered_data)
        
        if use_pca:
            filtered_data = self.apply_pca(n_components, filtered_data)
        
        return filtered_data
    
    def normalize_features(self, data=None):
        """Normalize feature columns."""
        if data is None:
            data = self.data.copy()
        
        if data.empty:
            print("Error: no data to normalize")
            return data
        
        feature_cols = [col for col in data.columns if col not in ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']]
        
        self.scaler.fit(data[feature_cols])
        data_scaled = data.copy()
        data_scaled[feature_cols] = self.scaler.transform(data[feature_cols])
        
        self.processed_data = data_scaled
        return data_scaled
    
    def set_date_range(self, start_date, end_date):
        """Update the date range for the loader."""
        self.start_date = start_date
        self.end_date = end_date
        self.data_start_date = self._get_adjusted_start_date(start_date, self.lookback_days)
        
        if self.data is not None and not self.data.empty:
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            
            self.data['date'] = pd.to_datetime(self.data['date'])
            mask = (self.data['date'] >= start) & (self.data['date'] <= end)
            self.data = self.data[mask].reset_index(drop=True)
            
            if self.benchmark is not None and not self.benchmark.empty:
                self.benchmark['date'] = pd.to_datetime(self.benchmark['date'])
                mask = (self.benchmark['date'] >= start) & (self.benchmark['date'] <= end)
                self.benchmark = self.benchmark[mask].reset_index(drop=True)
            
            for symbol in self.all_stock_data:
                stock_data = self.all_stock_data[symbol]
                stock_data['date'] = pd.to_datetime(stock_data['date'])
                mask = (stock_data['date'] >= start) & (stock_data['date'] <= end)
                self.all_stock_data[symbol] = stock_data[mask].reset_index(drop=True)
                
            print(f"Date range updated to {start_date} to {end_date}; data filtered")
        else:
            print(f"Date range updated to {start_date} to {end_date}; will apply on next load")

