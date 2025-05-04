import torch
import numpy as np
import pandas as pd
import faiss
import re
from sklearn.preprocessing import MinMaxScaler
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Tuple
from config import Config

class TimeSeriesAgent:
    def __init__(self):
        self.config = Config()
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.setup_models()
        self.setup_knowledge_base()

    def setup_models(self):
        """Initialize models with device optimization"""
        # Embedding model
        self.embedding_model = SentenceTransformer(
            self.config.EMBEDDING_MODEL,
            device=self.config.DEVICE
        )
        
        # LLM models for validation and optimization
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.LLM_MODEL)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.LLM_MODEL,
            torch_dtype=torch.float16 if 'cuda' in self.config.DEVICE else torch.float32
        ).to(self.config.DEVICE)

    def setup_knowledge_base(self):
        """Initialize FAISS index"""
        self.dimension = 384  # Dimension of MiniLM embeddings
        self.index = faiss.IndexFlatL2(self.dimension)
        self.pattern_db = []  # Stores (window, next_values) pairs

    def preprocess_series(self, series: pd.DataFrame, window_size: int, forecast_steps: int) -> Tuple[np.ndarray, np.ndarray, List]:
        """Normalize and create non-overlapping windows with timestamps"""
        values = self.scaler.fit_transform(series['value'].values.reshape(-1, 1)).flatten()
        windows, next_vals, timestamps = [], [], []

        # Use non-overlapping windows for cleaner forecasting
        stride = forecast_steps
        for i in range(0, len(values) - window_size - forecast_steps + 1, stride):
            windows.append(values[i:i+window_size])
            next_vals.append(values[i+window_size:i+window_size+forecast_steps])
            forecast_timestamps = series['timestamp'].iloc[i+window_size:i+window_size+forecast_steps]
            timestamps.extend(forecast_timestamps)

        return np.array(windows), np.array(next_vals), timestamps

    def build_knowledge_base(self, windows: np.ndarray, next_vals: np.ndarray):
        """Create embeddings for similar pattern retrieval"""
        for window, next_v in zip(windows, next_vals):
            self.pattern_db.append((window, next_v))
            self.index.add(self.embedding_model.encode([str(window)]))

    def retrieve_similar_patterns(self, query: np.ndarray, k: int = 3) -> List[np.ndarray]:
        """Find similar historical patterns and their subsequent values"""
        query_embed = self.embedding_model.encode([str(query)])
        _, indices = self.index.search(np.array(query_embed).astype('float32'), k)
        return [self.pattern_db[i][1] for i in indices[0] if i != -1]

    def validate_forecast(self, forecast: List[float], history: List[float]) -> bool:
        """Use LLM to validate forecast"""
        prompt = f"""Time Series Validation Task:
        Historical Window: {history[-12:]}
        Proposed Forecast: {forecast}

        Rules:
        1. Reject if >50% change between consecutive values
        2. Reject if negative values for positive-only series
        3. Reject if violates seasonal patterns

        Decision (Valid/Invalid):"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.config.DEVICE)
        output = self.model.generate(**inputs, max_new_tokens=3)
        decision = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return "valid" in decision.lower()

    def optimize_forecast(self, forecast: List[float], context: str) -> List[float]:
        """Use LLM to optimize forecast"""
        prompt = f"""Forecast Optimization:
        Original Forecast: {forecast}
        Context: {context}

        Adjustments Needed:
        1. Smooth extreme jumps
        2. Align with seasonal trends
        3. Maintain realistic growth rates

        Revised Forecast (comma-separated):"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.config.DEVICE)
        output = self.model.generate(**inputs, max_new_tokens=20)
        revised = self.tokenizer.decode(output[0], skip_special_tokens=True)

        try:
            return [float(x) for x in revised.split(",")][:self.config.DATASETS[dataset_name]['forecast_steps']]
        except:
            return forecast

    def predict(self, data: pd.DataFrame, dataset_name: str) -> Dict:
        """Complete prediction pipeline with validation and optimization"""
        try:
            if dataset_name not in self.config.DATASETS:
                raise ValueError(f"Unknown dataset: {dataset_name}")
                
            cfg = self.config.DATASETS[dataset_name]
            
            if data.empty:
                raise ValueError("Empty DataFrame provided")
                
            # Filter for target_id=0 only
            target_df = data[data[cfg['id_col']] == self.config.DEFAULT_TARGET_ID].sort_values(cfg['date_col'])
            
            # 1. Preprocess and create windows
            windows, next_vals, _ = self.preprocess_series(target_df, cfg['window_size'], cfg['forecast_steps'])
            
            # 2. Build knowledge base
            self.build_knowledge_base(windows, next_vals)
            
            # 3. Get last window for forecasting
            normalized = self.scaler.transform(target_df[cfg['target_col']].values.reshape(-1, 1)).flatten()
            last_window = normalized[-cfg['window_size']:]
            
            # 4. Retrieve similar patterns and generate initial forecast
            similar_patterns = self.retrieve_similar_patterns(last_window)
            initial_forecast = np.mean(similar_patterns, axis=0)
            
            # 5. Validate and optimize
            final_forecast = initial_forecast
            for attempt in range(self.config.MAX_RETRIES):
                if self.validate_forecast(final_forecast, last_window):
                    break
                final_forecast = self.optimize_forecast(final_forecast, f"Attempt {attempt}")
            
            # 6. Inverse transform
            final_forecast = self.scaler.inverse_transform(np.array(final_forecast).reshape(-1, 1)).flatten()
            
            return {
                'status': 'success',
                'forecast': final_forecast.tolist(),
                'last_window': last_window,
                'similar_patterns': [p.tolist() for p in similar_patterns]
            }
            
        except Exception as e:
            # Create fallback forecast
            fallback = [data[self.config.DATASETS[dataset_name]['target_col']].iloc[-1]] * self.config.DATASETS[dataset_name]['forecast_steps'] if not data.empty else [0] * self.config.DATASETS[dataset_name]['forecast_steps']
            return {
                'status': 'error',
                'error': str(e),
                'fallback': fallback
            }