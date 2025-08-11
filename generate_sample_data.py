
import pandas as pd
import numpy as np

def generate_sales(start='2020-01-01', periods=120, seed=42):
    np.random.seed(seed)
    dates = pd.date_range(start, periods=periods, freq='D')
    # trend + weekly seasonality + yearly seasonality + noise
    trend = np.linspace(50, 200, periods)
    weekly = 10 * np.sin(2 * np.pi * dates.dayofweek / 7)
    yearly = 20 * np.sin(2 * np.pi * dates.dayofyear / 365.25)
    noise = np.random.normal(0, 8, periods)
    sales = np.maximum(0, trend + weekly + yearly + noise).round().astype(int)
    df = pd.DataFrame({'date': dates, 'sales': sales})
    return df

if __name__ == '__main__':
    df = generate_sales()
    df.to_csv('synthetic_sales.csv', index=False)
    print('synthetic_sales.csv created with', len(df), 'rows')
