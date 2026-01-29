"""
Local CSV data loader.

Purpose: read local CSVs (Chinese or English headers), normalize to
date/open/high/low/close/volume/symbol/daily_return, and compute indicators
consistent with the project for direct use in the trading environment.

Conventions:
- Asset files: SYMBOL.xxx.csv (case-insensitive), symbol uses first segment (e.g., "ADS.DF.CSV" -> "ADS")
- Benchmark files contain keywords (default: GDAXI, STOXX50E) for benchmark creation
"""

import os
import re
import pandas as pd
import numpy as np
from typing import List, Dict, Optional

try:
    import talib as ta
except Exception:
    ta = None


CH_CN_TO_EN_COLS = {
    '日期': 'date', '时间': 'date',
    '开盘': 'open', '最高': 'high', '最低': 'low', '收盘': 'close',
    '成交量': 'volume', '量': 'volume',
}


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c: c for c in df.columns}
    lower = [str(c).strip().lower() for c in df.columns]
    mapping = {}
    for c, lc in zip(df.columns, lower):
        if lc in ('date', 'datetime', 'trade_date', 'tradedate', 'candle_begin_time', 'timestamp', 'time'):
            mapping[c] = 'date'
        elif lc in ('open', '开盘'):
            mapping[c] = 'open'
        elif lc in ('high', '最高'):
            mapping[c] = 'high'
        elif lc in ('low', '最低'):
            mapping[c] = 'low'
        elif lc in ('close', '收盘', 'adj close', 'adj_close', 'adjusted close'):
            mapping[c] = 'close'
        elif lc in ('volume', '成交量'):
            mapping[c] = 'volume'
        else:
            # Try direct Chinese mapping
            mapping[c] = CH_CN_TO_EN_COLS.get(str(c), c)
    df = df.rename(columns=mapping)
    # If still no date column, auto-detect 8-digit date column
    if 'date' not in df.columns:
        for c in df.columns:
            try:
                series = pd.to_numeric(df[c], errors='coerce')
                # If most values look like YYYYMMDD, treat as date column
                if series.notna().mean() > 0.9 and (series.dropna().astype(int).astype(str).str.len() == 8).mean() > 0.9:
                    df = df.rename(columns={c: 'date'})
                    break
            except Exception:
                continue
    # Ensure core columns exist; fill missing
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col not in df.columns:
            if col == 'volume':
                df[col] = 0
            elif col == 'close' and 'adj close' in lower:
                df[col] = df['adj close']
            else:
                df[col] = np.nan
    if 'date' not in df.columns:
        # Try index as date
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={'index': 'date'})
        else:
            raise ValueError('CSV missing date column and index is not date')
    # Normalize type (supports YYYY-MM-DD, YYYY/MM/DD, YYYYMMDD)
    # If 8-digit numeric, parse as YYYYMMDD first
    if df['date'].dtype in (np.int64, np.int32, np.float64, np.float32) or (
        df['date'].astype(str).str.len().median() == 8 and df['date'].astype(str).str.isnumeric().mean() > 0.8
    ):
        df['date'] = pd.to_datetime(df['date'].astype(str), format='%Y%m%d', errors='coerce')
    else:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['date'] = df['date'].dt.tz_localize(None)
    df = df.dropna(subset=['date']).sort_values('date').reset_index(drop=True)
    for c in ['open', 'high', 'low', 'close', 'volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def _calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # Daily return, volatility, momentum
    df['daily_return'] = df['close'].pct_change()
    df['volatility'] = df['daily_return'].rolling(20).std()
    df['momentum'] = df['close'].pct_change(10)

    # RSI 14
    try:
        if ta is not None:
            df['rsi_14'] = ta.RSI(df['close'].values, timeperiod=14)
        else:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            df['rsi_14'] = 100 - (100 / (1 + rs))
    except Exception:
        df['rsi_14'] = np.nan

    # MACD 12,26
    try:
        if ta is not None:
            macd, macdsignal, macdhist = ta.MACD(df['close'].values, 12, 26, 9)
            df['macd'] = macd
        else:
            exp1 = df['close'].ewm(span=12).mean()
            exp2 = df['close'].ewm(span=26).mean()
            df['macd'] = exp1 - exp2
    except Exception:
        df['macd'] = np.nan

    # Bollinger band width
    try:
        mean20 = df['close'].rolling(20).mean()
        std20 = df['close'].rolling(20).std()
        df['bb_width'] = (std20 * 2) / mean20
    except Exception:
        df['bb_width'] = np.nan

    # ATR14
    try:
        if ta is not None:
            df['atr_14'] = ta.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
        else:
            hl = df['high'] - df['low']
            hc = (df['high'] - df['close'].shift()).abs()
            lc = (df['low'] - df['close'].shift()).abs()
            tr = np.maximum(hl, np.maximum(hc, lc))
            df['atr_14'] = tr.rolling(14).mean()
    except Exception:
        df['atr_14'] = np.nan

    # Relative strength, volume change
    df['relative_strength'] = df['close'] / df['close'].rolling(50).mean()
    df['volume_change'] = df['volume'].pct_change()

    # Finalize: clip extreme returns, fill, remove inf
    df['daily_return'] = df['daily_return'].replace([np.inf, -np.inf], np.nan)
    df['daily_return'] = df['daily_return'].clip(lower=-0.5, upper=0.5)
    df = df.ffill().bfill()
    return df


class LocalCSVDataLoader:
    def __init__(self, start_date: str, end_date: str, symbols: List[str], features: List[str], data_dir: str, lookback_days: int = 90, benchmark_hint: Optional[str] = None):
        self.start_date = start_date
        self.end_date = end_date
        self.symbols = symbols
        self.features = features
        self.data_dir = data_dir
        self.lookback_days = lookback_days
        self.all_stock_data: Dict[str, pd.DataFrame] = {}
        self.benchmark: Optional[pd.DataFrame] = None
        self.data: Optional[pd.DataFrame] = None
        self.processed_data: Optional[pd.DataFrame] = None
        # Benchmark hint (from config data.benchmark)
        self.benchmark_hint = (benchmark_hint or '').upper().strip() or None

    def _list_csv_files(self) -> List[str]:
        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(f'data_dir not found: {self.data_dir}')
        files = [f for f in os.listdir(self.data_dir) if f.lower().endswith('.csv')]
        return files

    def _infer_symbol_from_filename(self, filename: str) -> str:
        base = os.path.basename(filename)
        name = re.split(r'\.|_', base)[0]
        return name.upper()

    def _is_benchmark_file(self, filename: str) -> bool:
        u = filename.upper()
        # Prefer benchmark hint from config
        if self.benchmark_hint:
            base = os.path.splitext(os.path.basename(filename))[0].upper()
            # Support common matches: contains, exact base, prefix match
            if (self.benchmark_hint in u) or (base == self.benchmark_hint) or base.startswith(self.benchmark_hint) or self.benchmark_hint.startswith(base):
                return True
        # Fallback to default keyword match
        return ('GDAXI' in u) or ('STOXX50E' in u)

    def load_data(self):
        files = self._list_csv_files()
        start = pd.to_datetime(self.start_date)
        end = pd.to_datetime(self.end_date)
        adj_start = start - pd.Timedelta(days=self.lookback_days)
        all_rows = []

        # Infer symbols from directory if not provided
        symset = set([s.upper() for s in (self.symbols or [])])
        if not symset:
            for f in files:
                if not self._is_benchmark_file(f):
                    symset.add(self._infer_symbol_from_filename(f))

        def _read_csv_robust(path: str) -> pd.DataFrame:
            # Try multiple encodings and separators
            encodings = ['utf-8', 'utf-8-sig', 'gbk', 'cp936', 'latin1']
            seps = [',', ';', '\t']
            last_err = None
            for enc in encodings:
                for sep in seps:
                    try:
                        df = pd.read_csv(path, sep=sep, encoding=enc, engine='python')
                        if df is not None and not df.empty:
                            return df
                    except Exception as e:
                        last_err = e
                        continue
            # Last error
            if last_err:
                raise last_err
            return pd.DataFrame()

        for f in files:
            path = os.path.join(self.data_dir, f)
            try:
                df = _read_csv_robust(path)
                df = _standardize_columns(df)
                sym = self._infer_symbol_from_filename(f)
                df['symbol'] = sym

                # Extend window for indicator computation
                df = df[(df['date'] >= adj_start) & (df['date'] <= end)].copy()
                if df.empty:
                    continue

                # Compute indicators
                df = _calc_indicators(df)

                # Trim back to trading window after indicators; keep aligned index
                df_trim = df[(df['date'] >= start) & (df['date'] <= end)].copy()
                if df_trim.empty:
                    continue

                if self._is_benchmark_file(f):
                    self.benchmark = df_trim.copy()
                    continue

                if sym in symset:
                    self.all_stock_data[sym] = df_trim.copy()
                    all_rows.append(df_trim.copy())
            except Exception as e:
                print(f"Failed to read {path}: {e}")

        if all_rows:
            self.data = pd.concat(all_rows, ignore_index=True)
        else:
            self.data = pd.DataFrame()

    def get_data_for_date_range(self, start_date=None, end_date=None, symbols=None, normalized=False, use_pca=False, n_components=20):
        # Return a filtered view of self.data; no forced normalization
        if self.data is None or self.data.empty:
            return pd.DataFrame()
        s = pd.to_datetime(start_date) if start_date else pd.to_datetime(self.start_date)
        e = pd.to_datetime(end_date) if end_date else pd.to_datetime(self.end_date)
        syms = set(symbols) if symbols else set(self.symbols)
        df = self.data[(self.data['date'] >= s) & (self.data['date'] <= e)].copy()
        if syms:
            df = df[df['symbol'].isin(syms)].copy()
        return df.reset_index(drop=True)

    def set_date_range(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date
        # Defer filtering to get_data_for_date_range




