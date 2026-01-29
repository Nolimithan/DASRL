# DASRL Long Mode: LIM-PPO Portfolio Optimization Framework

A reinforcement learning stock selection and portfolio optimization strategy based on the Long Intensity Measure (LIM) for long-only trading.

## Introduction

- Stock selection using LIM indicator
- Portfolio weight optimization via PPO reinforcement learning algorithm
- T+0 trading mechanism support
- Integer share trading compliant with actual market rules
- Multi-group comparison experiments (Top, Middle, Low groups)
- Comprehensive performance metrics and visualization tools

## Features

- **LIM-based Stock Selection**: Uses Long Intensity Measure to identify stocks with high short-term profit potential
- **Reinforcement Learning Optimization**: PPO algorithm for portfolio optimization balancing returns and risk
- **Multi-factor Feature Engineering**: Combines technical indicators, fundamental factors, and market features
- **Variance Penalty Reward Function**: Incorporates portfolio covariance as penalty to encourage diversification
- **Visualization**: Portfolio curves, holdings heatmaps, performance radar charts

## Project Structure

```
LIM_long_mode/
├── config/                 # Configuration files
│   ├── default.yaml        # Default config (512760.sh dataset)
│   └── sse50.yaml          # SSE50 dataset config
├── data/                   # Data directory
├── models/                 # Model definitions
│   └── ppo_agent.py        # PPO agent implementation
├── results/                # Results directory
├── utils/                  # Utility functions
│   ├── data_loader.py      # Data loading
│   ├── environment.py      # Trading environment
│   ├── lim_calculator.py   # LIM calculator
│   └── performance_metrics.py  # Performance evaluation
├── tools/                  # Auxiliary tools
│   └── setup_proxy.py      # Proxy configuration
├── visualization/          # Visualization modules
├── main.py                 # Main entry script
├── ablation_study.py       # Ablation study script
└── README.md               # Documentation
```

## Quick Start

### Data Preparation

1. Place stock historical data in the `data/` directory
2. Ensure data format meets requirements

### Training

Train with different stock group types:

```bash
# Train with Top group (highest LIM stocks)
python main.py --mode train --config config/default.yaml --group_type TOP

# Train with Middle group (median LIM stocks)
python main.py --mode train --config config/default.yaml --group_type MIDDLE

# Train with Low group (lowest LIM stocks)
python main.py --mode train --config config/default.yaml --group_type LOW
```

### Stock Group Types

The system supports three stock group types:
- **TOP**: Top group, selects n stocks with highest LIM values
- **MIDDLE**: Middle group, selects n stocks with median LIM values
- **LOW**: Low group, selects n stocks with lowest LIM values

Using different groups enables comparison experiments to validate LIM effectiveness.

### Testing

```bash
# Basic test
python main.py --mode test --config config/default.yaml --visualize

# Test with specific date range
python main.py --mode test --start_date 2024-04-06 --end_date 2025-04-06 --config config/default.yaml --visualize --group_type TOP
python main.py --mode test --start_date 2024-04-06 --end_date 2025-04-06 --config config/default.yaml --visualize --group_type MIDDLE
python main.py --mode test --start_date 2024-04-06 --end_date 2025-04-06 --config config/default.yaml --visualize --group_type LOW
```

### Ablation Study

```bash
# Run ablation study on Top group
python ablation_study.py --mode train --config config/default.yaml --group_type TOP
python ablation_study.py --mode test --start_date 2024-04-06 --end_date 2025-04-06 --group_type TOP

# Strategy comparison
python strategy_comparison.py --data_path data --config config/default.yaml --output_dir strategy_comparison_results --start_date 2024-04-06 --end_date 2025-04-06 --initial_capital 1000000
```

## Datasets

This framework supports two datasets for long-only strategies:

| Dataset | Description | Market | Config File |
|---------|-------------|--------|-------------|
| 512760.sh | China Semiconductor ETF constituents | A-share | `config/default.yaml` |
| SSE50 | Shanghai Stock Exchange 50 Index | A-share | `config/sse50.yaml` |

Data sources:
- China A-share data: via Tushare API
- Data is automatically cached locally after first download

## Configuration

The configuration file `config/default.yaml` contains:

- **Data config**: Start/end dates, stock symbols
- **Feature config**: Technical indicators, PCA parameters
- **LIM config**: LIM calculation parameters
- **Portfolio config**: Portfolio size, initial capital
- **Trading config**: Commission rate
- **RL config**: PPO algorithm parameters, state/action dimensions
- **Reward config**: Penalty weights
- **Experiment config**: Strategy types, group types (TOP/MIDDLE/LOW)

## Algorithm

### LIM Calculation

LIM (Long Intensity Measure) quantifies short-term profit potential:

1. **Daily Trading Return**: Mechanical strategy of buying at open, selling at close
   ```
   Wt = W0 * [Ct*(1-F)²/Ot]  (when Ct*(1-F)²/Ot ≥ 1)
   Wt = W0                   (when Ct*(1-F)²/Ot < 1)
   ```
   Where:
   - W0: Initial capital
   - Ct: Closing price on day t
   - Ot: Opening price on day t
   - F: Commission rate per transaction

2. **Cumulative Return**:
   ```
   LIMi = ∏ᵀₜ₌₁(Wt/W0)
   ```
   Only accumulates profitable trades

3. **Normalization**:
   ```
   LIM*ᵢ = (LIMᵢ/LIMₘₐₓ) × 100%
   ```

### PPO Algorithm

PPO (Proximal Policy Optimization) balances exploration and exploitation by constraining policy update magnitude.

### Reward Function

The reward function considers:
- Portfolio return rate
- Portfolio variance penalty
- Turnover penalty
- Maximum drawdown penalty

## Performance Metrics

Evaluation metrics include:
- Total Return
- Annualized Return
- Sharpe Ratio
- Sortino Ratio
- Maximum Drawdown
- Volatility
- Win Rate
- Turnover Rate
- Cost Ratio

## Command Line Arguments

```
usage: main.py [-h] [--mode {train,test,backtest}] [--config CONFIG] [--visualize]
               [--start_date START_DATE] [--end_date END_DATE]
               [--group_type {TOP,MIDDLE,LOW}]

DASRL Long Strategy

optional arguments:
  -h, --help            Show help message
  --mode {train,test,backtest}
                        Run mode: train, test, or backtest
  --config CONFIG       Configuration file path
  --visualize           Generate visualization results
  --start_date START_DATE
                        Test start date (format: YYYY-MM-DD)
  --end_date END_DATE   Test end date (format: YYYY-MM-DD)
  --group_type {TOP,MIDDLE,LOW}
                        Stock group type: TOP, MIDDLE, or LOW
```

## Dependencies

See `requirements.txt` for required packages.
