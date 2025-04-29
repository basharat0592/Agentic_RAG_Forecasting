import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def generate_ettm2_data(num_points=100):
    """Generate synthetic ETTm2 (Electricity Transformer Temperature) data"""
    base_date = datetime(2016, 7, 1)
    dates = [base_date + timedelta(minutes=15*i) for i in range(num_points)]
    
    # Base pattern with daily seasonality
    x = np.arange(num_points)
    daily = 2 * np.sin(2 * np.pi * x / (24*4))  # 24 hours * 4 (15-min intervals)
    
    # Weekly pattern
    weekly = 1.5 * np.sin(2 * np.pi * x / (24*4*7))
    
    # Random noise and trend
    noise = np.random.normal(0, 0.2, num_points)
    trend = 0.005 * x
    
    values = 30 + daily + weekly + trend + noise
    
    df = pd.DataFrame({'date': dates, 'OT': values.round(2)})
    return df

def generate_electricity_data(num_points=100):
    """Generate synthetic hourly electricity consumption data"""
    base_date = datetime(2020, 1, 1)
    dates = [base_date + timedelta(hours=i) for i in range(num_points)]
    
    # Daily seasonality
    x = np.arange(num_points)
    daily = 500 * np.sin(2 * np.pi * x / 24)
    
    # Weekly pattern
    weekly = 200 * np.sin(2 * np.pi * x / (24*7))
    
    # Random noise and trend
    noise = np.random.normal(0, 50, num_points)
    trend = 5 * x
    
    values = 3000 + daily + weekly + trend + noise
    
    df = pd.DataFrame({'date': dates, 'value': values.round(1)})
    return df

def generate_weather_data(num_points=100):
    """Generate synthetic hourly weather temperature data"""
    base_date = datetime(2020, 1, 1)
    dates = [base_date + timedelta(hours=i) for i in range(num_points)]
    
    # Diurnal cycle
    x = np.arange(num_points)
    daily = 5 * np.sin(2 * np.pi * x / 24)
    
    # Random noise and trend
    noise = np.random.normal(0, 0.5, num_points)
    trend = 0.02 * x
    
    values = 20 + daily + trend + noise
    
    df = pd.DataFrame({'date': dates, 't2m': values.round(1)})
    return df

def save_datasets():
    """Generate and save all sample datasets"""
    os.makedirs("data", exist_ok=True)
    
    print("Generating ETTm2 dataset...")
    ettm2 = generate_ettm2_data(500)  # 500 points (~5 days)
    ettm2.to_csv("data/ettm2.csv", index=False)
    
    print("Generating electricity dataset...")
    electricity = generate_electricity_data(168)  # 168 hours (1 week)
    electricity.to_csv("data/electricity.csv", index=False)
    
    print("Generating weather dataset...")
    weather = generate_weather_data(168)  # 168 hours (1 week)
    weather.to_csv("data/weather.csv", index=False)
    
    print("\nAll datasets generated successfully in 'data' directory:")
    print(f"- ettm2.csv: {len(ettm2)} rows (15-min interval temperature data)")
    print(f"- electricity.csv: {len(electricity)} rows (hourly consumption)")
    print(f"- weather.csv: {len(weather)} rows (hourly temperature)")

if __name__ == "__main__":
    save_datasets()