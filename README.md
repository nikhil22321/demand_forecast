
# AI-Powered Demand Forecasting & Inventory Optimization

**Timeline:** September 2024 – October 2024  
**Tech:** Python, Prophet, Pandas, Streamlit, Matplotlib, Statsmodels (ARIMA)

## What’s included
- `generate_sample_data.py` — creates `synthetic_sales.csv` (sample historical daily sales)
- `forecasting.py` — Prophet and ARIMA forecasting functions
- `inventory.py` — simple reorder suggestion utilities
- `app.py` — Streamlit app to run the demo
- `requirements.txt` — Python dependencies

## Quick start (Linux / macOS / Windows with Python)
1. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # macOS/Linux
   venv\Scripts\activate    # Windows (PowerShell)
   ```
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Generate sample data (optional):
   ```bash
   python generate_sample_data.py
   ```
4. Run the app:
   ```bash
   streamlit run app.py
   ```

## Notes
- `prophet` package (formerly `fbprophet`) can require a C++ compiler on Windows; if installation fails, use WSL or conda.
- ARIMA uses `statsmodels`; for automatic order selection you can integrate `pmdarima` (auto_arima).
- This is a demo skeleton. For production, add validation, retraining pipelines, model evaluation, and persistence.
