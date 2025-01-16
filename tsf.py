import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from math import sqrt

# Load the dataset
data = pd.read_csv('data_set.csv')

# Question 3(a) Part 1: Data Inspection
print("Data Info:")
print(data.info())
print("\nSummary Statistics:")
print(data.describe())

# Question 3(a) Part 2: Convert Data to Time Series
# Assuming columns are named 'Date' and 'Close'
# Convert 'Date' column to datetime with the correct format
data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y', dayfirst=True)
data.set_index('Date', inplace=True)
data = data.asfreq('D')  # Daily frequency
data.rename(columns={'Avg spending':'Close'}, inplace=True)

data['Close'] = data['Close'].interpolate(method='linear')  # Handle missing values

print("\nAfter interpolation, missing values:")
print(data.isnull().sum())

# Question 3(a) Part 3: Visualize the Time Series
plt.figure(figsize=(12, 6))
plt.plot(data['Close'], label='Closing Price')
plt.title('Closing Stock Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Question 3(b) Part 1: Decompose the Time Series
result = seasonal_decompose(data['Close'], model='additive')
result.plot()
plt.show()

# Question 3(b) Part 2: Test for Stationarity
adf_test = adfuller(data['Close'])
print("\nADF Test Results:")
print(f"ADF Statistic: {adf_test[0]}")
print(f"p-value: {adf_test[1]}")
print(f"Critical Values: {adf_test[4]}")

if adf_test[1] > 0.05:
    print("Time series is non-stationary. Differencing the data...")
    data['Close_diff'] = data['Close'].diff().dropna()
else:
    print("Time series is stationary.")

# Question 3(b) Part 3: Plot ACF and PACF
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.title('Autocorrelation Function (ACF)')
plt.stem(acf(data['Close'].dropna(), nlags=30))

plt.subplot(122)
plt.title('Partial Autocorrelation Function (PACF)')
plt.stem(pacf(data['Close'].dropna(), nlags=30))

plt.tight_layout()
plt.show()

# Question 3(c): Split into Train and Test Sets
train = data[:-60]  # All except the last 2 months
test = data[-60:]   # Last 2 months

# Fit ARIMA Model
model_arima = ARIMA(train['Close'], order=(1, 1, 1))
model_arima_fit = model_arima.fit()

# Forecast
forecast_arima = model_arima_fit.forecast(steps=60)

# Evaluate ARIMA Model
rmse_arima = sqrt(mean_squared_error(test['Close'], forecast_arima))
mape_arima = mean_absolute_percentage_error(test['Close'], forecast_arima)

print("\nARIMA Model Performance:")
print(f"RMSE: {rmse_arima}")
print(f"MAPE: {mape_arima}")

# Question 3(a): Exponential Smoothing Model
model_es = ExponentialSmoothing(train['Close'], trend='add', seasonal='add', seasonal_periods=12).fit()
forecast_es = model_es.forecast(steps=60)

# Evaluate Exponential Smoothing Model
rmse_es = sqrt(mean_squared_error(test['Close'], forecast_es))
mape_es = mean_absolute_percentage_error(test['Close'], forecast_es)

print("\nExponential Smoothing Model Performance:")
print(f"RMSE: {rmse_es}")
print(f"MAPE: {mape_es}")

# Improving Exponential Smoothing
model_es_improved = ExponentialSmoothing(train['Close'], trend='add', seasonal='mul', seasonal_periods=12).fit()
forecast_es_improved = model_es_improved.forecast(steps=60)

# Analyze Residuals
residuals = test['Close'] - forecast_es_improved
plt.figure(figsize=(12, 6))
plt.plot(residuals, label='Residuals')
plt.axhline(0, color='red', linestyle='--')
plt.title('Residuals of Improved Exponential Smoothing Model')
plt.legend()
plt.show()

# Evaluate Improved Model
rmse_es_improved = sqrt(mean_squared_error(test['Close'], forecast_es_improved))
mape_es_improved = mean_absolute_percentage_error(test['Close'], forecast_es_improved)

print("\nImproved Exponential Smoothing Model Performance:")
print(f"RMSE: {rmse_es_improved}")
print(f"MAPE: {mape_es_improved}")

# Forecasting Avg Spending for Next Month
next_month_forecast = model_es_improved.forecast(steps=30)
print("\nForecast for Next Month:")
print(next_month_forecast)
