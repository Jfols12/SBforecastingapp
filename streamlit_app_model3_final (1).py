
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from fredapi import Fred
from datetime import datetime

# --- Load Data ---
df = pd.read_csv("starbucks_financials_expanded.csv")
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# --- Get Live CPI from FRED and express as % (YoY inflation rate approximation) ---
fred = Fred(api_key="841ad742e3a4e4bb2f3fcb90a3d078fb")
cpi_series = fred.get_series('CPIAUCSL')
current_cpi = cpi_series.iloc[-1]
prev_cpi = cpi_series.iloc[-13]  # 12 months ago
cpi_percent = ((current_cpi - prev_cpi) / prev_cpi) * 100

st.title("📊 Starbucks Revenue Forecasting App")
st.write(f"### Current CPI (Year-over-Year): {cpi_percent:.2f}%")

# --- User Inputs for Forecast Variables ---
st.sidebar.header("User Forecast Inputs")
cpi_input = st.sidebar.slider("Adjusted CPI (%)", min_value=-5.0, max_value=5.0, value=2.0, step=0.1)
txn_input = st.sidebar.slider("Expected Transactions", 800, 1200, 1000, step=10)
mkt_input = st.sidebar.slider("Expected Marketing Spend ($M)", 300, 800, 500, step=10)

# --- Prepare ARIMAX Model 3 (CPI + Transactions + Marketing Spend) ---
endog = df['revenue']
exog = df[['cpi', 'transactions', 'marketing_spend']]

model = SARIMAX(endog, exog=exog, order=(1, 1, 1))
results = model.fit(disp=False)

# --- Create Exogenous Forecast Inputs ---
future_index = pd.date_range(start=df.index[-1] + pd.DateOffset(months=3), periods=4, freq='Q')
future_exog = pd.DataFrame({
    'cpi': [cpi_input] * 4,
    'transactions': [txn_input] * 4,
    'marketing_spend': [mkt_input] * 4
}, index=future_index)

forecast = results.get_forecast(steps=4, exog=future_exog)
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()

# --- Plot Forecasted vs Actual Revenue ---
st.subheader("📈 Forecasted vs Actual Revenue")
fig1, ax1 = plt.subplots()
df['revenue'].plot(ax=ax1, label='Actual Revenue')
forecast_mean.plot(ax=ax1, label='Forecasted Revenue', color='green')
ax1.fill_between(forecast_ci.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='green', alpha=0.3)
ax1.set_ylabel("Revenue")
ax1.legend()
st.pyplot(fig1)

# --- Plot Expenses vs Revenue ---
st.subheader("💸 Expenses vs Revenue Over Time")
fig2, ax2 = plt.subplots()
df[['revenue', 'expenses']].plot(ax=ax2)
ax2.set_ylabel("USD ($)")
st.pyplot(fig2)

# --- Comment on Expenses vs Revenue Relationship ---
correlation = df['revenue'].corr(df['expenses'])
st.markdown("### 🔍 Analysis of Expenses vs Revenue")
if correlation > 0.75:
    st.success(f"There is a strong positive correlation ({correlation:.2f}) between expenses and revenue.")
elif correlation > 0.5:
    st.info(f"There is a moderate positive correlation ({correlation:.2f}) between expenses and revenue.")
else:
    st.warning(f"The correlation ({correlation:.2f}) between expenses and revenue is weak or inconsistent.")

# --- AI Summary ---
st.subheader("🧠 AI Summary for Audit Committee")
summary = f"""
**Summary**: This ARIMAX model incorporates CPI (expressed as a percentage), transactions, and marketing spend. With a user-adjusted CPI of {cpi_input:.2f}%, {txn_input} expected transactions, and ${mkt_input}M in marketing, the model forecasts revenue for the next four quarters. The observed correlation between expenses and revenue is {correlation:.2f}, offering insight into the operational impact on top-line results.
"""
st.markdown(summary)
