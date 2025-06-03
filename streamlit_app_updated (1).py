
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from fredapi import Fred
import datetime

# --- Load Data ---
df = pd.read_csv("starbucks_financials_expanded.csv")
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# --- Get Live CPI from FRED ---
fred = Fred(api_key="841ad742e3a4e4bb2f3fcb90a3d078fb")
latest_cpi = fred.get_series_latest_release('CPIAUCSL').iloc[-1]
st.write(f"### 📈 Current CPI (FRED): {latest_cpi:.2f}")

# --- User Inputs for CPI and Transactions ---
st.sidebar.header("User Input")
cpi_input = st.sidebar.number_input("Enter expected CPI (%)", value=3.0, step=0.1)
txn_input = st.sidebar.number_input("Enter expected Transactions", value=1000, step=1)

# Adjusted values to apply to forecast
adjusted_cpi_value = latest_cpi * (1 + cpi_input / 100)

# --- Forecast with Transactions + CPI (ARIMAX) ---
endog = df['revenue']
exog = df[['transactions']].copy()
exog['cpi'] = df['cpi']

model1 = SARIMAX(endog, exog=exog, order=(1,1,1))
results1 = model1.fit(disp=False)

# Prepare exog for forecasting
forecast_exog = pd.DataFrame({
    'transactions': [txn_input] * 4,
    'cpi': [adjusted_cpi_value] * 4
}, index=pd.date_range(start=df.index[-1] + pd.DateOffset(months=3), periods=4, freq='Q'))

forecast = results1.get_forecast(steps=4, exog=forecast_exog)
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()

# --- Plot Forecasted vs Actual Revenue ---
st.subheader("Forecasted vs Actual Revenue")
fig1, ax1 = plt.subplots()
df['revenue'].plot(ax=ax1, label='Actual')
forecast_mean.plot(ax=ax1, label='Forecast', color='green')
ax1.fill_between(forecast_ci.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='green', alpha=0.3)
ax1.legend()
st.pyplot(fig1)

# --- Plot Revenue vs Expenses ---
st.subheader("Expenses vs Revenue")
fig2, ax2 = plt.subplots()
df[['revenue', 'expenses']].plot(ax=ax2)
st.pyplot(fig2)

# --- ARIMA with Expenses + CPI ---
exog2 = df[['expenses']].copy()
exog2['cpi'] = df['cpi']

model2 = SARIMAX(endog, exog=exog2, order=(1,1,1))
results2 = model2.fit(disp=False)

st.subheader("ARIMAX Model with Expenses")
st.write("#### Coefficients:")
st.write(results2.params)

st.write("#### P-values:")
st.write(results2.pvalues)

exp_coef = results2.params['expenses']
exp_pval = results2.pvalues['expenses']
if exp_pval < 0.05:
    st.success(f"The expenses variable is statistically significant (p = {exp_pval:.4f}) with a coefficient of {exp_coef:.4f}.")
else:
    st.warning(f"The expenses variable is NOT statistically significant (p = {exp_pval:.4f}), though its coefficient is {exp_coef:.4f}.")

# --- AI Summary ---
st.subheader("AI Summary for Audit Committee")
st.markdown("""
> **Summary**: Based on our ARIMAX analysis, Starbucks revenue is sensitive to changes in both CPI and expenses. The expenses variable shows [statistical/non-statistical] significance, suggesting [a potential risk/an observed trend] in revenue recognition. The audit committee should be aware that elevated CPI or rising operational expenses may inflate revenue forecasts, potentially overstating financial performance.
""")
