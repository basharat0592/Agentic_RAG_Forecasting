import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from typing import Dict, List

class ResultsAnalyzer:
    def __init__(self, agent):
        self.agent = agent
        self.results_dir = "results"
        os.makedirs(f"{self.results_dir}/figures", exist_ok=True)
        os.makedirs(f"{self.results_dir}/tables", exist_ok=True)
        
        # Configure matplotlib style
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams['figure.autolayout'] = True

    def plot_results(self, data: pd.Series, result: Dict, dataset_name: str):
        """Generate comprehensive result visualization"""
        if 'forecast' not in result:
            print("Error: No forecast in results")
            return None
            
        cfg = self.agent.config.DATASETS[dataset_name]
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Historical data
        ax.plot(data.index[-3*cfg['window_size']:], 
               data.values[-3*cfg['window_size']:],
               label='Historical Data', linewidth=2, color='#1f77b4')
        
        # Forecast
        freq = cfg['freq']
        forecast_dates = pd.date_range(
            start=data.index[-1],
            periods=len(result['forecast'])+1,
            freq=freq
        )[1:]
        
        ax.plot(forecast_dates, result['forecast'],
               'ro--', label='Forecast', linewidth=2, markersize=8)
        
        ax.set_title(f'{dataset_name.upper()} Forecast\nWindow: {cfg["window_size"]} points | Freq: {freq}',
                   fontsize=14)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.legend()
        plt.xticks(rotation=45)
        
        plot_path = f"{self.results_dir}/figures/{dataset_name}_forecast.png"
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path

    def calculate_metrics(self, actual: pd.Series, forecast: List[float]) -> Dict:
        """Calculate forecast accuracy metrics"""
        actual_values = actual.values[-len(forecast):]
        metrics = {
            'MAE': np.mean(np.abs(actual_values - forecast)),
            'RMSE': np.sqrt(np.mean((actual_values - forecast)**2)),
            'MAPE': np.mean(np.abs((actual_values - forecast)/actual_values)) * 100,
            'R2': 1 - np.sum((actual_values - forecast)**2)/np.sum((actual_values - np.mean(actual_values))**2)
        }
        return metrics

    def save_metrics(self, metrics: Dict, dataset_name: str):
        """Save metrics to CSV and LaTeX"""
        df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
        
        csv_path = f"{self.results_dir}/tables/{dataset_name}_metrics.csv"
        df.to_csv(csv_path)
        
        latex_path = f"{self.results_dir}/tables/{dataset_name}_metrics.tex"
        df.style.format(precision=4).to_latex(latex_path)
        
        return csv_path, latex_path

    def generate_report(self, data: pd.Series, result: Dict, dataset_name: str):
        """Generate all results and metrics"""
        plot_path = self.plot_results(data, result, dataset_name)
        
        if result['status'] == 'success':
            actual = data.iloc[-len(result['forecast']):]
            metrics = self.calculate_metrics(actual, result['forecast'])
            csv_path, latex_path = self.save_metrics(metrics, dataset_name)
            
            return {
                'plot': plot_path,
                'metrics': metrics,
                'csv_path': csv_path,
                'latex_path': latex_path
            }
        return {'plot': plot_path}