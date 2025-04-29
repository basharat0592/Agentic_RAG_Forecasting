import torch
import numpy as np
import pandas as pd
import faiss
import re
from sklearn.preprocessing import MinMaxScaler
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
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
        """Initialize models with CPU optimization"""
        # Embedding model
        self.embedding_model = SentenceTransformer(
            self.config.EMBEDDING_MODEL,
            device=self.config.DEVICE
        )
        
        # Seq2Seq model for forecasting
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.LLM_MODEL)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.LLM_MODEL,
            torch_dtype=torch.float32
        ).to(self.config.DEVICE)

    def setup_knowledge_base(self):
        """Initialize FAISS index"""
        self.dimension = 384  # Dimension of MiniLM embeddings
        self.index = faiss.IndexFlatL2(self.dimension)
        self.knowledge_base = []

    def preprocess_series(self, series: pd.Series, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Normalize and create sliding windows"""
        values = series.values.reshape(-1, 1)
        normalized = self.scaler.fit_transform(values).flatten()
        
        windows = []
        for i in range(len(normalized) - window_size):
            windows.append(normalized[i:i+window_size])
            
        return np.array(windows), normalized[-window_size:]

    def build_knowledge_base(self, windows: np.ndarray):
        """Create embeddings for similar pattern retrieval"""
        text_windows = [f"TS:{' '.join(map(str, window))}" for window in windows]
        embeddings = self.embedding_model.encode(text_windows, show_progress_bar=False)
        self.index.add(np.array(embeddings).astype('float32'))
        self.knowledge_base.extend(text_windows)

    def retrieve_similar_patterns(self, query: np.ndarray, k: int = 3) -> List[str]:
        """Find similar historical patterns"""
        query_embed = self.embedding_model.encode([f"TS:{' '.join(map(str, query))}"])
        _, indices = self.index.search(np.array(query_embed).astype('float32'), k)
        return [self.knowledge_base[i] for i in indices[0] if i != -1]

    def generate_forecast(self, prompt: str, forecast_steps: int) -> List[float]:
        """Generate forecast with constrained output"""
        inputs = self.tokenizer(
            prompt,
            max_length=self.config.MAX_SEQ_LENGTH,
            truncation=True,
            return_tensors="pt"
        ).to(self.config.DEVICE)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=50,
            num_beams=5,
            early_stopping=True
        )
        
        pred_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        numbers = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", pred_text)
        return [float(num) for num in numbers[:forecast_steps]]

    def predict(self, data: pd.DataFrame, dataset_name: str) -> Dict:
        """Complete prediction pipeline"""
        try:
            if dataset_name not in self.config.DATASETS:
                raise ValueError(f"Unknown dataset: {dataset_name}")
                
            cfg = self.config.DATASETS[dataset_name]
            
            if data.empty:
                raise ValueError("Empty DataFrame provided")
                
            if cfg['target_col'] not in data.columns:
                available_cols = ", ".join(data.columns)
                raise ValueError(f"Target column '{cfg['target_col']}' not found. Available columns: {available_cols}")
            
            series = data[cfg['target_col']]
            
            if len(series) < cfg['window_size'] + cfg['forecast_steps']:
                raise ValueError(f"Not enough data. Need at least {cfg['window_size'] + cfg['forecast_steps']} points, got {len(series)}")
            
            # 1. Preprocess
            windows, last_window = self.preprocess_series(series, cfg['window_size'])
            
            # 2. Build knowledge base
            self.build_knowledge_base(windows)
            
            # 3. Retrieve similar patterns
            similar = self.retrieve_similar_patterns(last_window)
            
            # 4. Create enriched prompt
            stats = {
                'mean': np.mean(last_window),
                'std': np.std(last_window),
                'min': np.min(last_window),
                'max': np.max(last_window)
            }
            
            prompt = f"""
            Time series forecasting task ({dataset_name}):
            - Frequency: {cfg['freq']}
            - Stats: mean={stats['mean']:.2f}, std={stats['std']:.2f}
            - Last {cfg['window_size']} normalized values: {' '.join(map(str, last_window))}
            
            Similar historical patterns:
            {''.join(similar[:2])}
            
            Predict next {cfg['forecast_steps']} values. Respond ONLY with numbers separated by spaces.
            """
            
            # 5. Generate forecast
            pred_norm = self.generate_forecast(prompt, cfg['forecast_steps'])
            pred_denorm = self.scaler.inverse_transform(
                np.array(pred_norm).reshape(-1, 1)
            ).flatten()
            
            return {
                'status': 'success',
                'forecast': pred_denorm.tolist(),
                'last_window': last_window,
                'similar_patterns': similar
            }
            
        except Exception as e:
            # Create fallback forecast
            fallback = [data[self.config.DATASETS[dataset_name]['target_col']].iloc[-1]] * self.config.DATASETS[dataset_name]['forecast_steps'] if not data.empty else [0] * self.config.DATASETS[dataset_name]['forecast_steps']
            return {
                'status': 'error',
                'error': str(e),
                'fallback': fallback
            }