
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

# --- Get Live CPI from FRED ---
fred = Fred(api_key="841ad742e3a4e4bb2f3fcb90a3d078fb")
latest_cpi = fred.get_series_latest_release('CPIAUCSL').iloc[-1]
prev_cpi = fred.get_series('CPIAUCSL').iloc[-2]
cpi_change_pct = ((latest_cpi - prev_cpi) / prev_cpi) * 100

st.title("📊 Starbucks Revenue Forecasting App")
st.write(f"### Live CPI Change: {cpi_change_pct:.2f}%")

# --- User Inputs for Forecast Variables ---
st.sidebar.header("User Forecast Inputs")
cpi_input = st.sidebar.number_input("Adjust CPI (%)", min_value=-10.0, max_value=10.0, value=0.0, step=0.1)
txn_input = st.sidebar.slider("Expected Transactions", 800, 1200, 1000, step=10)
mkt_input = st.sidebar.slider("Expected Marketing Spend ($M)", 300, 800, 500, step=10)

# --- Adjusted Forecast Inputs ---
adjusted_cpi = df['cpi'].copy() * (1 + cpi_input / 100)
adj_txn = txn_input
adj_mkt = mkt_input

# --- Prepare ARIMAX Model 3 (CPI + Transactions + Marketing Spend) ---
endog = df['revenue']
exog = pd.DataFrame({
    'adj_cpi': adjusted_cpi,
    'transactions': df['transactions'],
    'marketing_spend': df['marketing_spend']
})

model = SARIMAX(endog, exog=exog, order=(1, 1, 1))
results = model.fit(disp=False)

# --- Create Exogenous Forecast Inputs ---
future_index = pd.date_range(start=df.index[-1] + pd.DateOffset(months=3), periods=4, freq='Q')
future_exog = pd.DataFrame({
    'adj_cpi': [adjusted_cpi.iloc[-1]] * 4,
    'transactions': [adj_txn] * 4,
    'marketing_spend': [adj_mkt] * 4
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
**Summary**: The updated ARIMAX model (Model 3) incorporates CPI, transactions, and marketing spend. A CPI change of {cpi_input:.2f}%, combined with {txn_input} transactions and ${mkt_input}M in marketing spend, directly influences revenue forecasts. The expenses correlation with revenue is {correlation:.2f}, indicating a {"strong" if correlation > 0.75 else "moderate" if correlation > 0.5 else "weak"} relationship, and should be considered in audit evaluations.
"""
st.markdown(summary)
