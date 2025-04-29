# Agentic RAG Time Series Forecasting

A hybrid forecasting system combining retrieval-augmented generation (RAG) with transformer models for time series prediction.

## Features

- Supports 5 datasets (M4, Electricity, ETTm2, BigData22, Weather)
- FAISS-based pattern retrieval
- BART-based forecasting
- Automated visualization and metrics
- Graceful fallback mechanisms

## Installation

```bash
git clone https://github.com/basharat0592/agentic-time-series-forecasting.git
cd agentic-time-series-forecasting
pip install -r requirements.txt
```

## Usage

```bash
python run_demo.py
```

Follow prompts to:
1. Select dataset (1-5)
2. View loaded data
3. Generate forecasts
4. See results and metrics

## Configuration

Edit `config.py` to:
- Add/modify datasets
- Change window sizes
- Adjust forecast steps
- Switch models

## File Structure

```
├── datasets_csv/       # Input data
├── results/           # Outputs
├── config.py          # Settings
├── data_loader.py     # Data processing  
├── agent.py           # Forecasting core
├── analysis.py        # Visualization
└── run_demo.py        # Main interface
```

## Requirements

- Python 3.8+
- pandas, numpy
- scikit-learn
- PyTorch
- transformers
- sentence-transformers
- faiss-cpu
- matplotlib

## Outputs

- Forecast plots: `results/figures/`
- Metrics: `results/tables/` (CSV & LaTeX)

## License

MIT
