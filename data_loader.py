import pandas as pd
import numpy as np
import os
from typing import Dict, Tuple
from config import Config

class DataLoader:
    def __init__(self):
        self.config = Config()
        os.makedirs("data", exist_ok=True)

    def _load_csv(self, path: str, date_col: str) -> pd.DataFrame:
        """Load CSV file with memory optimization"""
        try:
            return pd.read_csv(path, parse_dates=[date_col])
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None

    def _preprocess_data(self, df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
        """Clean and prepare the data for target_id=0"""
        # Filter for target_id=0 only
        if cfg['id_col'] in df.columns:
            df = df[df[cfg['id_col']] == self.config.DEFAULT_TARGET_ID]
        
        # Set datetime index
        df = df.set_index(cfg['date_col']).sort_index()
        
        # Handle column names
        if cfg['target_col'] not in df.columns:
            if 'OT' in df.columns:  # ETTm2 case
                cfg['target_col'] = 'OT'
            elif 'Temp' in df.columns:  # Weather case
                cfg['target_col'] = 'Temp'
            elif 'value' in df.columns:
                cfg['target_col'] = 'value'
        
        # Resample to consistent frequency and clean
        return df[[cfg['target_col']]].resample(cfg['freq']).mean().interpolate().ffill().bfill()

    def load_dataset(self, name: str) -> Tuple[pd.DataFrame, Dict]:
        """Load and preprocess specified dataset for target_id=0"""
        if name not in self.config.DATASETS:
            raise ValueError(f"Unknown dataset: {name}")
        
        cfg = self.config.DATASETS[name]
        df = self._load_csv(cfg['path'], cfg['date_col'])
        
        if df is None:
            return None, cfg
        
        processed = self._preprocess_data(df, cfg)
        return processed, cfg

    def get_sample_series(self, df: pd.DataFrame, target_col: str) -> pd.Series:
        """Get the target time series"""
        return df[target_col]