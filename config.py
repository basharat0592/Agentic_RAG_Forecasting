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
            'freq': 'M',
            'id_col': 'id'
        },
        'electricity': {
            'path': 'datasets_csv/gluonts_electricity.csv',
            'window_size': 24,  # 24 hours
            'forecast_steps': 12,
            'target_col': 'value',
            'date_col': 'timestamp',
            'freq': 'H',
            'id_col': 'id'
        },
        'ettm2': {
            'path': 'datasets_csv/ettm2_monthly.csv',
            'window_size': 12,  # 1 year
            'forecast_steps': 6,
            'target_col': 'OT',
            'date_col': 'date',
            'freq': 'M',
            'id_col': 'id'
        },
        'bigdata22': {
            'path': 'datasets_csv/bigdata22_monthly.csv',
            'window_size': 12,  # 1 year
            'forecast_steps': 6,
            'target_col': 'value',
            'date_col': 'timestamp',
            'freq': 'M',
            'id_col': 'id'
        },
        'weather': {
            'path': 'datasets_csv/weather_monthly.csv',
            'window_size': 12,  # 1 year
            'forecast_steps': 6,
            'target_col': 'Temp',
            'date_col': 'Date',
            'freq': 'M',
            'id_col': 'id'
        }
    }
    
    # Model configuration
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    LLM_MODEL = "microsoft/phi-2"  # Changed to phi-2
    VALIDATOR_MODEL = "microsoft/phi-2"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # System configuration
    MAX_RETRIES = 3
    MIN_VALUE = 0.01
    MAX_SEQ_LENGTH = 512
    DEFAULT_TARGET_ID = 0  # Default series ID to use