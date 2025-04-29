import torch

class Config:
    # Dataset configurations
    DATASETS = {
        'm4': {
            'path': 'datasets_csv/m4_monthly.csv',
            'window_size': 12,  # 1 year of monthly data
            'forecast_steps': 6,
            'target_col': 'value',
            'date_col': 'timestamp',
            'freq': 'M'
        },
        'electricity': {
            'path': 'datasets_csv/gluonts_electricity.csv',
            'window_size': 24,  # 24 hours
            'forecast_steps': 12,
            'target_col': 'value',
            'date_col': 'timestamp',
            'freq': 'H'
        },
        'ettm2': {
            'path': 'datasets_csv/ettm2_monthly.csv',
            'window_size': 12,  # 1 year
            'forecast_steps': 6,
            'target_col': 'OT',
            'date_col': 'date',
            'freq': 'M'
        },
        'bigdata22': {
            'path': 'datasets_csv/bigdata22_monthly.csv',
            'window_size': 12,  # 1 year
            'forecast_steps': 6,
            'target_col': 'value',
            'date_col': 'timestamp',
            'freq': 'M'
        },
        'weather': {
            'path': 'datasets_csv/weather_monthly.csv',
            'window_size': 12,  # 1 year
            'forecast_steps': 6,
            'target_col': 'Temp',
            'date_col': 'Date',
            'freq': 'M'
        }
    }
    
    # Model configuration
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    LLM_MODEL = "facebook/bart-base"
    DEVICE = "cpu"
    
    # System configuration
    MAX_RETRIES = 3
    MIN_VALUE = 0.01
    MAX_SEQ_LENGTH = 512