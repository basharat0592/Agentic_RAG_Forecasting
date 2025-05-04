from data_loader import DataLoader
from agent import TimeSeriesAgent
from analysis import ResultsAnalyzer
import time

def main():
    print("=== Agentic RAG Time Series Forecasting ===")
    print("Initializing system...")
    start_time = time.time()
    
    # Initialize components
    data_loader = DataLoader()
    agent = TimeSeriesAgent()
    analyzer = ResultsAnalyzer(agent)
    
    # Available datasets
    datasets = {
        '1': 'm4',
        '2': 'electricity',
        '3': 'ettm2', 
        '4': 'bigdata22',
        '5': 'weather'
    }
    
    print("\nAvailable datasets:")
    for k, v in datasets.items():
        print(f"{k}. {v.upper()}")
    
    choice = input("\nSelect dataset (1-5): ").strip()
    dataset_name = datasets.get(choice, 'm4')
    
    print(f"\nLoading {dataset_name} dataset (series ID={agent.config.DEFAULT_TARGET_ID})...")
    try:
        data, cfg = data_loader.load_dataset(dataset_name)
        
        if data is None:
            print(f"Error: Failed to load dataset {dataset_name}")
            return
            
        series = data_loader.get_sample_series(data, cfg['target_col'])
        print(f"\nSelected series with {len(series)} points")
        print(series.head())
        
        print(f"\nGenerating {cfg['forecast_steps']}-step forecast using Agentic RAG...")
        result = agent.predict(data, dataset_name)
        
        print("\n=== Results ===")
        if result['status'] == 'success':
            print(f"Forecast: {result['forecast']}")
            report = analyzer.generate_report(series, result, dataset_name)
            
            if report and 'metrics' in report:
                print("\nPerformance Metrics:")
                for metric, value in report['metrics'].items():
                    print(f"{metric}: {value:.4f}")
                
                print(f"\nVisualization saved to: {report['plot']}")
                print(f"Metrics saved to: {report['csv_path']}")
        else:
            print(f"Error: {result['error']}")
            print(f"Fallback forecast: {result['fallback']}")
            
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
        
    print(f"\nTotal execution time: {time.time()-start_time:.2f} seconds")

if __name__ == "__main__":
    main()