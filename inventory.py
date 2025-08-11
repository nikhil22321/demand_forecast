
import pandas as pd
import numpy as np

def reorder_suggestions(forecast_df, current_stock, lead_time_days=7, safety_factor=1.2):
    # forecast_df: dataframe with 'ds' and 'yhat' for next N days
    demand_during_lead = forecast_df.head(lead_time_days)['yhat'].sum()
    suggested_order = np.maximum(0, (demand_during_lead * safety_factor) - current_stock)
    return int(np.ceil(suggested_order))

def days_of_stock_left(current_stock, daily_demand_forecast):
    # daily_demand_forecast: first value of forecast_df['yhat'] (next day)
    if daily_demand_forecast <= 0:
        return float('inf')
    return current_stock / daily_demand_forecast
