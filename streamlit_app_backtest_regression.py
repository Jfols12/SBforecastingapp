
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import LinearRegression
from fredapi import Fred
from datetime import datetime

# --- Load Data ---
df = pd.read_csv("starbucks_financials_expanded.csv")
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# --- Get Live CPI from FRED ---
fred = Fred(api_key="841ad742e3a4e4bb2f3fcb90a3d078fb")
cpi_series = fred.get_series('CPIAUCSL')
current_cpi = cpi_series.iloc[-1]
prev_cpi = cpi_series.iloc[-13]
cpi_percent = ((current_cpi - prev_cpi) / prev_cpi) * 100

st.title("📊 Starbucks Revenue Forecasting App (Backtest)")
st.write(f"### Current CPI (Year-over-Year): {cpi_percent:.2f}%")

# --- User Inputs ---
st.sidebar.header("User Inputs")
cpi_input = st.sidebar.slider("Adjusted CPI (%)", min_value=-5.0, max_value=5.0, value=2.0, step=0.1)
txn_input = st.sidebar.slider("Expected Transactions", 800, 1200, 1000, step=10)
mkt_input = st.sidebar.slider("Expected Marketing Spend ($M)", 300, 800, 500, step=10)

# --- Define forecast window (2023–2024) ---
backtest_period = df.loc["2023-01-01":]
train_data = df.loc[:'2022-12-31']

# --- Prepare ARIMAX model using CPI + transactions + marketing_spend ---
endog_train = train_data['revenue']
exog_train = train_data[['cpi', 'transactions', 'marketing_spend']]

model = SARIMAX(endog_train, exog=exog_train, order=(1, 1, 1))
results = model.fit(disp=False)

# --- Construct exog for the backtest period using user input overrides ---
backtest_index = backtest_period.index
exog_backtest = pd.DataFrame({
    'cpi': [cpi_input] * len(backtest_index),
    'transactions': [txn_input] * len(backtest_index),
    'marketing_spend': [mkt_input] * len(backtest_index)
}, index=backtest_index)

forecast = results.get_prediction(start=backtest_index[0], end=backtest_index[-1], exog=exog_backtest)
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()
actual = df['revenue']

# --- Plot Full Historical + Forecasted Revenue ---
st.subheader("📉 Full Revenue Timeline with Backtest Forecast Overlay")
fig1, ax1 = plt.subplots()
actual.plot(ax=ax1, label='Actual Revenue', color='blue')
forecast_mean.plot(ax=ax1, label='Backtested Forecast (2023–2024)', color='green')
ax1.fill_between(forecast_ci.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='green', alpha=0.3)
ax1.set_ylabel("Revenue")
ax1.set_title("Revenue with Backtested Forecast (2023–2024)")
ax1.legend()
st.pyplot(fig1)

# --- Forecast Error Analysis ---
errors = backtest_period['revenue'] - forecast_mean
mae = np.mean(np.abs(errors))
mape = np.mean(np.abs(errors / backtest_period['revenue'])) * 100

st.subheader("📊 Forecast Error Analysis")
st.write(f"**Mean Absolute Error (MAE):** ${mae:,.2f}")
st.write(f"**Mean Absolute Percentage Error (MAPE):** {mape:.2f}%")

# --- Expenses Regression Model from Revenue ---
st.subheader("💼 Regression-Based Expense Prediction")
X = df[['revenue']]
y = df['expenses']
model_reg = LinearRegression().fit(X, y)
df['predicted_expenses'] = model_reg.predict(X)

# --- Plot Actual Revenue, Actual Expenses, and Predicted Expenses ---
fig2, ax2 = plt.subplots()
df['revenue'].plot(ax=ax2, label='Actual Revenue', color='blue')
df['expenses'].plot(ax=ax2, label='Actual Expenses', color='red')
df['predicted_expenses'].plot(ax=ax2, label='Predicted Expenses (from Revenue)', linestyle='--', color='green')
ax2.set_ylabel("USD ($)")
ax2.set_title("Actual vs Predicted Expenses Using Revenue Regression")
ax2.legend()
st.pyplot(fig2)

# --- Correlation Analysis ---
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
**Summary**: The ARIMAX model was backtested on 2023–2024 using CPI ({cpi_input:.2f}%), transactions ({txn_input}), and marketing spend (${mkt_input}M). MAPE was {mape:.2f}%, with a correlation of {correlation:.2f} between revenue and expenses. A regression model predicts expenses using revenue, helping to identify variance and potential misreporting of costs.
"""
st.markdown(summary)
