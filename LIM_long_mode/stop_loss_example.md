# ATR Dynamic Stop-Loss Strategy Guide

This document explains how to backtest and compare the ATR (Average True Range) dynamic stop-loss strategy against a traditional drawdown-based stop-loss.

## Overview

Key characteristics of the ATR dynamic stop-loss strategy:

1. **Adaptive volatility** - adjusts the stop-loss threshold based on market volatility
2. **Trend-aware** - tightens stops in downtrends and loosens in uptrends
3. **Absolute safety bound** - caps maximum drawdown to prevent extreme losses

## Usage

### 1. Basic comparison

Run the following command to compare the baseline drawdown stop-loss with the ATR dynamic stop-loss:

```bash
python main.py --mode test --compare_stop_loss --start_date 2022-01-01 --end_date 2022-12-31
```

This will:
- Run both stop-loss strategies
- Compare metrics such as return, max drawdown, and Sharpe ratio
- Generate comparison plots
- Export detailed backtest data

### 2. Parameter optimization

Run the following command to test different parameter combinations and find the best ATR stop-loss parameters:

```bash
python main.py --mode test --optimize_stop_loss --start_date 2022-01-01 --end_date 2022-12-31
```

This tests multiple preset parameter sets, including:
- Different ATR windows (5/10/14/20/30 days)
- Different volatility ranges
- Whether to use trend adjustment

The system reports metrics for all parameter sets and selects the best one by Sharpe ratio.

### 3. Compare and optimize together

```bash
python main.py --mode test --compare_stop_loss --optimize_stop_loss --start_date 2022-01-01 --end_date 2022-12-31
```

### 4. With visualization

```bash
python main.py --mode test --compare_stop_loss --visualize --start_date 2022-01-01 --end_date 2022-12-31
```

## Parameter details

The ATR stop-loss strategy has three key parameters:

1. **atr_window**: ATR window length (default 14)
   - Short windows (5-10 days): more sensitive, faster response
   - Long windows (20-30 days): smoother, less noise

2. **volatility_range**: Volatility scaling range, formatted as (min, max)
   - Default (0.5, 2.0) means:
     - In low volatility, stop-loss can shrink to 50% of baseline
     - In high volatility, stop-loss can expand to 200% of baseline

3. **trend_adjustment**: Whether to enable trend adjustment (default True)
   - Adjusts the stop-loss threshold based on recent trend direction
   - Tightens stops in clear downtrends

## Output interpretation

The comparison report includes:

1. **Total return comparison**
2. **Max drawdown comparison**
3. **Sharpe ratio comparison**
4. **Stop-loss trigger count comparison**

It also produces four comparison charts:
- Portfolio value comparison
- Drawdown comparison
- Cash ratio comparison
- Cumulative return comparison

## Best practices

1. **Test on a short window first** - use 1-3 months to verify configuration
2. **Test across market regimes** - bull, bear, and range-bound markets
3. **Focus on risk metrics** - improve max drawdown and Sharpe, not just total return
4. **Validate before live trading** - use the most recent data for final checks
5. **Re-optimize periodically** - market dynamics change; consider quarterly tuning

## Advanced usage

To customize ATR stop-loss behavior, adjust:

1. ATR calculation method: `ATRStopLossTester._check_stop_loss()`
2. Trend logic: also in `_check_stop_loss()`
3. Parameter sets: `parameter_sets` in `main.py`