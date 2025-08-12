
import streamlit as st
import pandas as pd
import numpy as np
from forecasting import prophet_forecast, arima_forecast, ensemble_forecast
from inventory import reorder_suggestions, days_of_stock_left
import matplotlib.pyplot as plt

import streamlit as st
st.set_page_config(page_title='Demand Forecast & Inventory', layout='wide')

st.title("Demand Forecast App")
st.write("Hello! This is a test run.")



st.title('AI-Powered Demand Forecasting & Inventory Optimization')

# Upload or use sample data
uploaded = st.file_uploader('Upload historical sales CSV (columns: date,sales)', type=['csv'])
if uploaded is not None:
    df = pd.read_csv(uploaded, parse_dates=['date'])
else:
    st.info('Using synthetic sample data. You can upload your own CSV.')
    df = pd.read_csv('synthetic_sales.csv', parse_dates=['date'])

df = df.sort_values('date').reset_index(drop=True)
st.sidebar.header('Forecast Settings')
periods = st.sidebar.number_input('Forecast days', min_value=7, max_value=365, value=30)
model_choice = st.sidebar.selectbox('Model', ['Prophet', 'ARIMA', 'Ensemble'])
lead_time = st.sidebar.number_input('Lead time (days)', min_value=1, max_value=30, value=7)
current_stock = st.sidebar.number_input('Current stock (units)', min_value=0, value=100)

# prepare for Prophet
df_prophet = df.rename(columns={'date':'ds', 'sales':'y'})[['ds','y']]

with st.spinner('Running forecast...'):
    if model_choice == 'Prophet':
        forecast = prophet_forecast(df_prophet, periods=periods)
    elif model_choice == 'ARIMA':
        series = df.set_index('date')['sales']
        forecast = arima_forecast(series, periods=periods)
    else:
        forecast = ensemble_forecast(df_prophet, periods=periods, weights=(0.5, 0.5))

       

# show forecast table
st.subheader('Forecast (next {} days)'.format(periods))
st.dataframe(forecast.head(periods))

# plot historical + forecast
fig, ax = plt.subplots(figsize=(10,4))
ax.plot(df['date'], df['sales'], label='historical')
ax.plot(forecast['ds'], forecast['yhat'], label='forecast')
ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], alpha=0.2)
ax.set_xlabel('date'); ax.set_ylabel('sales'); ax.legend()
st.pyplot(fig)

# inventory suggestions
suggest = reorder_suggestions(forecast, current_stock=current_stock, lead_time_days=lead_time)
st.subheader('Inventory recommendations')
st.write(f'Current stock: {current_stock} units')
st.write(f'Suggested order quantity (to cover next {lead_time} days with safety factor): **{suggest} units**')
daily_next = float(forecast.head(1)['yhat'].iloc[0])
st.write(f'Estimated daily demand (next day): {daily_next:.2f}')
st.write(f'Estimated days of stock left: {days_of_stock_left(current_stock, daily_next):.1f} days')
