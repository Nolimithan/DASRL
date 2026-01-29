"""
Data loader for domestic market data and indicators.
"""

import os
import pandas as pd
import numpy as np
import talib as ta
import tushare as ts
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from datetime import datetime, timedelta


class DataLoader:
    """Data loader for stock data and indicators."""
    
    def __init__(self, start_date, end_date, symbols, features=None, data_dir='data', lookback_days=90):
        """Initialize the data loader."""
        self.start_date = start_date
        self.end_date = end_date
        self.symbols = symbols
        self.data_dir = data_dir
        self.lookback_days = lookback_days
        
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
        
        self.ts_token = 'b5d06c2132acd3329c1f771f4b3bf6d2b5b8ea05ea385981707d39d3'
        self._init_tushare()
        
        self.all_stock_data = {}
    
    def _get_adjusted_start_date(self, start_date, lookback_days):
        """Compute adjusted start date with lookback."""
        try:
            date_obj = datetime.strptime(start_date, '%Y-%m-%d')
            adjusted_date = date_obj - timedelta(days=lookback_days)
            return adjusted_date.strftime('%Y-%m-%d')
        except Exception as e:
            print(f"Date adjustment error: {e}. Using original date.")
            return start_date
    
    def _init_tushare(self):
        """Initialize Tushare API."""
        ts.set_token(self.ts_token)
        self.pro = ts.pro_api()
        print("Tushare API initialized")
    
    def _convert_symbol_to_tushare(self, symbol):
        """Convert symbol to Tushare format."""
        if symbol in ['512760', '159801', '512480']:
            return f"{symbol}.SH" if symbol.startswith('5') else f"{symbol}.SZ"
        
        if symbol.isdigit() and len(symbol) == 6:
            if symbol.startswith('6'):
                return f"{symbol}.SH"
            elif symbol.startswith('0') or symbol.startswith('3'):
                return f"{symbol}.SZ"
            elif symbol.startswith('688'):
                return f"{symbol}.SH"
            else:
                return f"{symbol}.SZ"
        
        return symbol
    
    def _get_stock_data_from_tushare(self, symbol):
        """Fetch stock data from Tushare."""
        ts_code = self._convert_symbol_to_tushare(symbol)
        start_date_fmt = self.data_start_date.replace('-', '')
        end_date_fmt = self.end_date.replace('-', '')
        
        try:
            if ts_code.endswith('.SH') or ts_code.endswith('.SZ'):
                if ts_code.startswith('51') or ts_code.startswith('15'):
                    print(f"Fetching ETF data: {ts_code}")
                    df = self.pro.fund_daily(ts_code=ts_code, 
                                            start_date=start_date_fmt, 
                                            end_date=end_date_fmt)
                else:
                    df = self.pro.daily(ts_code=ts_code, 
                                       start_date=start_date_fmt, 
                                       end_date=end_date_fmt)
                
                if df is not None and not df.empty:
                    df.rename(columns={
                        'trade_date': 'date', 
                        'vol': 'volume',
                        'amount': 'amount'
                    }, inplace=True)
                    
                    df['date'] = pd.to_datetime(df['date'])
                    
                    df['symbol'] = symbol
                    
                    if 'volume' not in df.columns and 'vol' in df.columns:
                        df['volume'] = df['vol']
                    elif 'volume' not in df.columns:
                        df['volume'] = 0
                    
                    df = df.sort_values('date', ascending=True).reset_index(drop=True)
                    return df
            
            print(f"Warning: unable to fetch {symbol} from Tushare")
            return None
                
        except Exception as e:
            print(f"Error fetching {symbol} data: {e}")
            return None
    
    def _get_benchmark_from_tushare(self):
        """Fetch benchmark index data from Tushare."""
        start_date_fmt = self.data_start_date.replace('-', '')
        end_date_fmt = self.end_date.replace('-', '')
        
        benchmark_symbols = ['512760.SH', '159801.SZ', '512480.SH']
        
        for ts_code in benchmark_symbols:
            try:
                print(f"Trying benchmark: {ts_code}")
                df = self.pro.fund_daily(ts_code=ts_code, 
                                        start_date=start_date_fmt, 
                                        end_date=end_date_fmt)
                
                if df is not None and not df.empty:
                    df.rename(columns={'trade_date': 'date'}, inplace=True)
                    df['date'] = pd.to_datetime(df['date'])
                    df['daily_return'] = df['close'].pct_change()
                    df = df.sort_values('date', ascending=True).reset_index(drop=True)
                    print(f"Benchmark fetched: {ts_code}, points: {len(df)}")
                    return df
            except Exception as e:
                print(f"Error fetching benchmark {ts_code}: {e}")
        
        try:
            print("Falling back to SSE index benchmark...")
            df = self.pro.index_daily(ts_code='000001.SH', 
                                     start_date=start_date_fmt, 
                                     end_date=end_date_fmt)
            if df is not None and not df.empty:
                df.rename(columns={'trade_date': 'date'}, inplace=True)
                df['date'] = pd.to_datetime(df['date'])
                df['daily_return'] = df['close'].pct_change()
                df = df.sort_values('date', ascending=True).reset_index(drop=True)
                print(f"SSE benchmark fetched, points: {len(df)}")
                return df
        except Exception as e:
            print(f"Error fetching SSE benchmark: {e}")
        
        print("Warning: unable to fetch benchmark data; using synthetic benchmark")
        return None
    
    def load_data(self):
        """Load stock data, compute indicators, and create benchmark."""
        print(f"Loading data for {len(self.symbols)} symbols...")
        
        os.makedirs(self.data_dir, exist_ok=True)
        
        all_data = []
        success_count = 0
        
        for symbol in self.symbols:
            print(f"Fetching data for {symbol}...")
            df = self._get_stock_data_from_tushare(symbol)
            
            if df is not None and not df.empty:
                df = df.sort_values('date').reset_index(drop=True)
                
                df['daily_return'] = df['close'].pct_change()
                
                df = self.calculate_technical_indicators(df)
                
                all_data.append(df)
                
                self.all_stock_data[symbol] = df
                
                success_count += 1
            else:
                print(f"Skipping {symbol}: no data")
        
        if all_data:
            self.data = pd.concat(all_data, ignore_index=True)
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
        self.benchmark = self._get_benchmark_from_tushare()
        
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
        
        return df
    
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
    
    def apply_pca(self, n_components=20, data=None):
        """Apply PCA for dimensionality reduction."""
        if data is None:
            if self.processed_data is not None:
                data = self.processed_data.copy()
            else:
                print("Error: no processed data available for PCA")
                return None
        
        if data.empty:
            print("Error: no data available for PCA")
            return data
        
        feature_cols = [col for col in data.columns if col not in ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']]
        
        pca = PCA(n_components=min(n_components, len(feature_cols)))
        pca_result = pca.fit_transform(data[feature_cols])
        
        pca_df = pd.DataFrame(
            pca_result, 
            columns=[f'PC{i+1}' for i in range(pca_result.shape[1])]
        )
        
        pca_df['date'] = data['date'].values
        pca_df['symbol'] = data['symbol'].values
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in data.columns:
                pca_df[col] = data[col].values
        
        if 'daily_return' in data.columns:
            pca_df['daily_return'] = data['daily_return'].values
        
        return pca_df
    
    def get_data_for_date_range(self, start_date=None, end_date=None, symbols=None, normalized=True, use_pca=False, n_components=20):
        """Get data within a date range and symbol set."""
        if self.data is None or self.data.empty:
            print("Error: no data loaded")
            return pd.DataFrame()
        
        start = pd.to_datetime(start_date) if start_date else pd.to_datetime(self.start_date)
        end = pd.to_datetime(end_date) if end_date else pd.to_datetime(self.end_date)
        
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
    
    def calculate_market_features(self):
        """Calculate market-level features from benchmark data."""
        if self.benchmark is None or self.benchmark.empty:
            raise ValueError("Benchmark data not loaded; cannot compute market features")
        
        market_data = self.benchmark.copy()
        
        market_data = self.calculate_technical_indicators(market_data)
        
        market_data['momentum_5d'] = market_data['close'].pct_change(5)
        market_data['momentum_10d'] = market_data['close'].pct_change(10)
        market_data['momentum_20d'] = market_data['close'].pct_change(20)
        
        market_data['volatility_20d'] = market_data['daily_return'].rolling(window=20).std()
        
        if 'RSI_14' in market_data.columns:
            market_data['market_sentiment'] = (market_data['RSI_14'] - 50) / 50
        
        market_data = market_data.dropna().reset_index(drop=True)
        
        return market_data
    
    def set_date_range(self, start_date, end_date):
        """Update the date range for the loader."""
        self.start_date = start_date
        self.end_date = end_date
        
        if self.data is not None and not self.data.empty:
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            
            mask = (self.data['date'] >= start) & (self.data['date'] <= end)
            self.data = self.data[mask].reset_index(drop=True)
            
            if self.benchmark is not None and not self.benchmark.empty:
                mask = (self.benchmark['date'] >= start) & (self.benchmark['date'] <= end)
                self.benchmark = self.benchmark[mask].reset_index(drop=True)
            
            for symbol in self.all_stock_data:
                stock_data = self.all_stock_data[symbol]
                mask = (stock_data['date'] >= start) & (stock_data['date'] <= end)
                self.all_stock_data[symbol] = stock_data[mask].reset_index(drop=True)
                
            print(f"Date range updated to {start_date} to {end_date}; data filtered")
        else:
            print(f"Date range updated to {start_date} to {end_date}; will apply on next load") 

