# DASRL Short Mode: SIM-PPO Portfolio Optimization Framework

A reinforcement learning stock selection and portfolio optimization strategy based on the Short Intensity Measure (SIM) for short-selling trading.

## Introduction

- Stock selection using SIM indicator
- Portfolio weight optimization via PPO reinforcement learning algorithm
- T+0 trading mechanism support
- Integer share trading compliant with actual market rules
- Multi-group comparison experiments (Top, Middle, Low groups)
- Comprehensive performance metrics and visualization tools

## Features

- **SIM-based Stock Selection**: Uses Short Intensity Measure to identify stocks with high short-selling potential
- **Reinforcement Learning Optimization**: PPO algorithm for portfolio optimization balancing returns and risk
- **Multi-factor Feature Engineering**: Combines technical indicators, fundamental factors, and market features
- **Variance Penalty Reward Function**: Incorporates portfolio covariance as penalty to encourage diversification
- **Visualization**: Portfolio curves, holdings heatmaps, performance radar charts

## Project Structure

```
SIM_short_mode/
├── config/                 # Configuration files
│   ├── default.yaml        # Default config
│   └── nasdaq_semiconductor.yaml  # SOXX dataset config
├── data/                   # Data directory
├── models/                 # Model definitions
│   └── ppo_agent.py        # PPO agent implementation
├── results/                # Results directory
├── utils/                  # Utility functions
│   ├── data_loader.py      # Data loading
│   ├── environment.py      # Trading environment
│   ├── sim_calculator.py   # SIM calculator
│   └── performance_metrics.py  # Performance evaluation
├── visualization/          # Visualization modules
├── main.py                 # Main entry script
├── ablation_study.py       # Ablation study script
└── README.md               # Documentation
```

## Quick Start

### Data Preparation

1. Place stock historical data in the `data/` directory
2. NASDAQ data is automatically downloaded via yfinance and cached locally

### Training

Train short-selling strategy with NASDAQ Semiconductor stocks:

```bash
# Train with Top group (highest SIM stocks)
python main.py --mode train --config config/nasdaq_semiconductor.yaml --group_type TOP

# Train with Middle group (median SIM stocks)
python main.py --mode train --config config/nasdaq_semiconductor.yaml --group_type MIDDLE

# Train with Low group (lowest SIM stocks)
python main.py --mode train --config config/nasdaq_semiconductor.yaml --group_type LOW
```

### Stock Group Types

The system supports three stock group types:
- **TOP**: Top group, selects n stocks with highest SIM values (best short candidates)
- **MIDDLE**: Middle group, selects n stocks with median SIM values
- **LOW**: Low group, selects n stocks with lowest SIM values

Using different groups enables comparison experiments to validate SIM effectiveness.

### Testing

```bash
# Basic test
python main.py --mode test --config config/nasdaq_semiconductor.yaml --visualize --group_type TOP

# Test with specific date range
python main.py --mode test --start_date 2024-04-06 --end_date 2025-04-06 --config config/nasdaq_semiconductor.yaml --visualize --group_type TOP
python main.py --mode test --start_date 2024-04-06 --end_date 2025-04-06 --config config/nasdaq_semiconductor.yaml --visualize --group_type MIDDLE
python main.py --mode test --start_date 2024-04-06 --end_date 2025-04-06 --config config/nasdaq_semiconductor.yaml --visualize --group_type LOW
```

### Ablation Study

```bash
# Run ablation study on Top group
python ablation_study.py --mode train --config config/nasdaq_semiconductor.yaml --group_type TOP
python ablation_study.py --mode test --start_date 2024-04-06 --end_date 2025-04-06 --group_type TOP
```

## Dataset

This framework uses NASDAQ semiconductor stocks for short-selling strategies:

| Dataset | Description | Market | Config File |
|---------|-------------|--------|-------------|
| SOXX | NASDAQ Semiconductor ETF constituents | US | `config/nasdaq_semiconductor.yaml` |

Data sources:
- NASDAQ data: via yfinance API
- Data is automatically cached locally after first download

## Configuration

The configuration file `config/nasdaq_semiconductor.yaml` contains:

- **Data config**: Start/end dates, stock symbols
- **Feature config**: Technical indicators, PCA parameters
- **SIM config**: SIM calculation parameters
- **Portfolio config**: Portfolio size, initial capital
- **Trading config**: Commission rate
- **RL config**: PPO algorithm parameters, state/action dimensions
- **Reward config**: Penalty weights
- **Experiment config**: Strategy types, group types (TOP/MIDDLE/LOW)

## Algorithm

### SIM Calculation

SIM (Short Intensity Measure) quantifies short-selling profit potential:

1. **Daily Short Trading Return**: Mechanical strategy of selling at open, buying at close
   ```
   Wt = W0 * [Ot/(Ct*(1-F)²)]  (when Ot/(Ct*(1-F)²) ≥ 1)
   Wt = W0                     (when Ot/(Ct*(1-F)²) < 1)
   ```
   Where:
   - W0: Initial capital
   - Ct: Closing price on day t
   - Ot: Opening price on day t
   - F: Commission rate per transaction

2. **Cumulative Return**:
   ```
   SIMi = ∏ᵀₜ₌₁(Wt/W0)
   ```
   Only accumulates profitable short trades (opening price > closing price)

3. **Normalization**:
   ```
   SIM*ᵢ = (SIMᵢ/SIMₘₐₓ) × 100%
   ```

### PPO Algorithm

PPO (Proximal Policy Optimization) balances exploration and exploitation by constraining policy update magnitude.

### Reward Function

The reward function considers:
- Short portfolio return rate
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

DASRL Short Strategy

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
