Here's the revised `README.md` that reflects your new implementation:

```markdown
# Agentic RAG Time Series Forecasting

A hybrid forecasting system combining retrieval-augmented generation (RAG) with transformer models for time series prediction, featuring validation and optimization agents.

## Key Features

- **Multi-Agent Architecture**:
  - Forecaster: Retrieves similar patterns using FAISS
  - Validator: Ensures forecast consistency using phi-2
  - Optimizer: Refines forecasts based on domain rules

- **Supported Datasets**:
  - M4 (Monthly)
  - Electricity (Hourly)
  - ETTm2 (Monthly)
  - BigData22 (Monthly)
  - Weather (Monthly)

- **Advanced Capabilities**:
  - Pattern retrieval with MiniLM embeddings
  - LLM-based forecast validation
  - Context-aware optimization
  - Automatic metric calculation (MAE, RMSE, MAPE, R²)
  - Comprehensive visualization

## Installation

```bash
git clone https://github.com/basharat0592/Agentic_RAG_Forecasting.git
cd Agentic_RAG_Forecasting
pip install -r requirements.txt
```

## Usage

1. Run the demo:
```bash
python run_demo.py
```

2. Follow the prompts to:
   - Select dataset (1-5)
   - View loaded data
   - Generate forecasts
   - See results and metrics

## Configuration

Edit `config.py` to:
- Change default target series ID
- Adjust window sizes and forecast steps
- Modify validation thresholds
- Switch between different LLM models

## System Architecture

```
├── datasets_csv/       # Input datasets
├── results/            # Output directory
│   ├── figures/        # Forecast visualizations
│   └── tables/         # Performance metrics
├── agent.py            # Core forecasting logic
├── analysis.py         # Visualization & metrics
├── config.py           # System configuration
├── data_loader.py      # Data preprocessing
└── run_demo.py         # Command-line interface
```

## Technical Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.7+ (for GPU acceleration)
- 8GB+ RAM (16GB recommended for larger datasets)

## License

MIT License

## Citation

If you use this system in your research, please cite:

```bibtex
@misc{agenticragforecasting,
  title = {Agentic RAG Time Series Forecasting System},
  author = {Basharat Hussain},
  year = {2025},
  url = {https://github.com/basharat0592/Agentic_RAG_Forecasting}
}
```

## Troubleshooting

**Q:** Getting CUDA out of memory errors  
**A:** Reduce batch size in `config.py` or use smaller window sizes

**Q:** Forecasts seem unrealistic  
**A:** Adjust validation thresholds in the validator agent

**Q:** Slow performance  
**A:** Enable GPU acceleration or reduce number of retrieved patterns