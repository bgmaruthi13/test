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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

# Load the dataset
data = pd.read_csv('data_set.csv')

# Data Preparation
data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y', dayfirst=True)
data.set_index('Date', inplace=True)
data = data.asfreq('D')  # Daily frequency
data.rename(columns={'Avg spending': 'Close'}, inplace=True)
data['Close'] = data['Close'].interpolate(method='linear')  # Handle missing values

# Test for Stationarity
adf_test = adfuller(data['Close'])
if adf_test[1] > 0.05:
    print("The series is non-stationary. Differencing the data...")
    data['Close_diff'] = data['Close'].diff().dropna()
    series_to_model = data['Close_diff'].dropna()
else:
    print("The series is stationary.")
    series_to_model = data['Close']

# Manual ARIMA Parameter Search
best_aic = np.inf
best_order = None
best_model = None

# Set ranges for p, d, q
p_range = 3
d_range = 2
q_range = 3

p = 0
while p <= p_range:
    d = 0
    while d <= d_range:
        q = 0
        while q <= q_range:
            try:
                # Fit ARIMA model
                model = ARIMA(series_to_model, order=(p, d, q))
                fitted_model = model.fit()
                aic = fitted_model.aic  # Calculate AIC
                print(f"ARIMA({p}, {d}, {q}) - AIC: {aic}")
                
                # Update the best model if the current one has a lower AIC
                if aic < best_aic:
                    best_aic = aic
                    best_order = (p, d, q)
                    best_model = fitted_model
            except Exception as e:
                print(f"ARIMA({p}, {d}, {q}) failed to fit. Error: {e}")
            q += 1
        d += 1
    p += 1

# Best Model Summary
if best_model:
    print("\nBest ARIMA Model:")
    print(f"Order: {best_order}")
    print(f"AIC: {best_aic}")
    print(best_model.summary())


### Define and explain the components of a time series decomposition.  
Time series decomposition splits a series into three main components: **trend**, **seasonality**, and **residuals/noise**.  
1. **Trend**: The long-term movement in the data, such as a consistent increase or decrease over time.  
2. **Seasonality**: Recurring patterns or cycles at regular intervals, e.g., monthly sales spikes during holidays.  
3. **Residuals/Noise**: Random, irregular fluctuations not explained by trend or seasonality.  
Additive decomposition assumes the series is the sum of these components, while multiplicative decomposition assumes the series is their product.  
For example, in retail sales, seasonal peaks may overlay an upward trend, with residual noise due to unexpected events.  

---

### Why is stationarity important in time series analysis? Provide methods for testing stationarity, also explain how does differencing help in making a time series stationary?  
Stationarity ensures that a time series has constant mean, variance, and autocorrelation over time, making it predictable and suitable for models like ARIMA.  
**Testing Stationarity**:  
1. **Visual Inspection**: Look for trends or varying variance in plots.  
2. **Dickey-Fuller Test**: A statistical test where a low p-value (< 0.05) indicates stationarity.  
3. **KPSS Test**: Complements the Dickey-Fuller test by checking for stationarity around a trend.  
**Differencing**: Subtracting consecutive values (e.g., \( y_t - y_{t-1} \)) removes trends, stabilizing the mean. Repeated differencing may be needed for more complex trends.  

---

### How will you determine the order of a moving average process? Explain.  
The order (\(q\)) of a moving average (MA) process is determined by analyzing the **Autocorrelation Function (ACF)**.  
1. In an MA(q) process, the ACF cuts off after lag \(q\), showing significant autocorrelation up to \(q\) lags and near-zero beyond it.  
2. Example: If the ACF has spikes at lags 1 and 2 but not beyond, the process may be MA(2).  
3. The Partial Autocorrelation Function (PACF) is not relevant for identifying \(q\) as it captures direct relationships.  
Correct identification of \(q\) ensures the model captures random noise components effectively.  

---

### Explain the main components of a time series. How do these components affect the analysis of a time series? Provide examples where applicable.  
The main components are:  
1. **Trend**: The overall direction of the series, e.g., increasing global temperatures. Ignoring it can bias analysis.  
2. **Seasonality**: Repeating patterns, e.g., summer tourism surges. Failing to account for it affects forecasting accuracy.  
3. **Noise**: Random variation, e.g., daily stock price fluctuations. Needs to be minimized for better model performance.  
Proper analysis involves isolating these components to understand their impact, e.g., using decomposition to forecast holiday sales spikes.  

---

### Define stationarity in the context of time series analysis. Why is it important for a time series to be stationary? Describe the process of differencing and how it helps in achieving stationarity.  
Stationarity means the statistical properties of a time series (mean, variance, autocorrelation) remain constant over time.  
It is crucial because most time series models (e.g., ARIMA) assume stationarity for reliable parameter estimation and forecasting.  
**Differencing**: Subtracting adjacent observations removes trends and stabilizes the mean, e.g.,  
\( Y_t' = Y_t - Y_{t-1} \).  
Additional differencing may handle higher-order trends or seasonal effects. Stationarity ensures model predictions focus on meaningful patterns, not unstable trends.  

---

### What is the role of moving average in time series analysis? Explain how the Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) are used in identifying the order of an ARIMA model.  
Moving averages smooth time series by averaging values over a window to reduce noise. They also form the basis of MA models in ARIMA, capturing short-term dependencies.  
**ACF and PACF** are diagnostic tools for identifying ARIMA parameters:  
- **ACF**: Used for determining \(q\) (MA order); significant spikes indicate the number of lags with autocorrelation.  
- **PACF**: Used for determining \(p\) (AR order); significant spikes show direct relationships with previous terms.  
By analyzing ACF and PACF, suitable ARIMA(\(p, d, q\)) orders can be selected.  

---

### Explain the concept of seasonality in time series analysis. How can seasonality be detected and accounted for in a time series model? Provide examples where applicable.  
Seasonality refers to recurring patterns in data at fixed intervals, e.g., monthly, quarterly, or yearly cycles.  
**Detection**:  
1. Visualize data to identify regular peaks or troughs.  
2. Seasonal decomposition (e.g., STL) separates seasonal patterns.  
3. ACF reveals peaks at seasonal lags.  
**Accounting for Seasonality**:  
1. Add seasonal components to models (e.g., SARIMA or seasonal exponential smoothing).  
2. Perform seasonal differencing (\(y_t - y_{t-s}\)).  
For example, modeling energy consumption requires accounting for winter heating spikes.  

---

### What are the key differences between the Autocorrelation Function (ACF) and the Partial Autocorrelation Function (PACF) in time series analysis? Explain their respective roles in identifying patterns within a time series.  
1. **ACF**: Measures the correlation between a series and its lagged values, including indirect relationships.  
   - Used to identify MA(\(q\)) order where autocorrelation cuts off.  
2. **PACF**: Measures the direct correlation of a series with its lagged values, excluding intermediate terms.  
   - Used to identify AR(\(p\)) order where partial autocorrelation cuts off.  
For example, in AR(1), PACF has a single spike at lag 1, while ACF decays geometrically.  

---

### What is the difference between additive and multiplicative time series models? In what situations would you choose one over the other? Provide examples to illustrate your answer.  
1. **Additive Model**: Assumes components combine linearly: \( Y_t = T_t + S_t + R_t \).  
   - Suitable when seasonal variations are constant over time (e.g., monthly sales increase by a fixed amount).  
2. **Multiplicative Model**: Assumes components interact multiplicatively: \( Y_t = T_t \times S_t \times R_t \).  
   - Used when seasonal variations scale with the trend (e.g., tourism grows proportionally with population).  
Choose based on seasonality behavior; for example, apply a multiplicative model for exponential sales growth during peak seasons.  

                                                                               
#!/usr/bin/env python
# coding: utf-8

# Import the necessary libraries.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from pylab import rcParams
from datetime import datetime, timedelta
from statsmodels.tsa.stattools import adfuller, pacf, acf
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.graphics.gofplots import qqplot
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings('ignore')

# Configure matplotlib inline
get_ipython().run_line_magic('matplotlib', 'inline')

# Read the data set in a Time Series with proper Time frequency or period.
df = pd.read_csv('MaunaLoa.csv', parse_dates=['Year-Month'], index_col='Year-Month')
df.head(15)

# Plot the Time Series Data.
rcParams['figure.figsize'] = 15, 8
df.plot()
plt.grid()

# Plot a boxplot to understand the variation of Carbon Dioxide in parts per million with respect to months across years.
sns.boxplot(x=df.index.month, y=df['CO2 ppm'])
plt.grid()

# Plot a graph of monthly Carbon Dioxide in parts per million across years.
monthly_co2ppm_across_years = pd.pivot_table(df, values='CO2 ppm', columns=df.index.year, index=df.index.month_name())
monthly_co2ppm_across_years.plot()
plt.grid()
plt.legend(loc='best')

# Decompose the Time Series to understand the various components.
decomposition = seasonal_decompose(df, model='additive')
decomposition.plot()

# Stationarity test
sns.boxplot(x=df.index.year, y=df['CO2 ppm'])
plt.grid()
observations = df.values
test_result = adfuller(observations)
test_result

# Applying differencing
df_diff = df.diff(periods=1).dropna()
observations = df_diff.values
test_result = adfuller(observations)
test_result

# Check the ACF and PACF of the training data.
plot_acf(df, lags=30)
plot_pacf(df)
plot_acf(df_diff)
plot_pacf(df_diff)

# Train-Test split
train_end = datetime(1978, 12, 1)
test_end = datetime(1980, 12, 1)
train = df[:train_end]
test = df[train_end + timedelta(days=1):test_end]

# Selecting an order of ARIMA model for data with the lowest Akaike Information Criteria (AIC).
p = q = range(0, 4)
d = range(1, 2)
pdq = list(itertools.product(p, d, q))
print('Parameter combinations for the Model:')
for i in range(1, len(pdq)):
    print(f'Model: {pdq[i]}')

dfObj1 = pd.DataFrame(columns=['param', 'AIC'])

for param in pdq:
    try:
        mod = ARIMA(train, order=param)
        results_Arima = mod.fit()
        print(f'ARIMA{param} - AIC:{results_Arima.aic}')
        dfObj1 = dfObj1.append({'param': param, 'AIC': results_Arima.aic}, ignore_index=True)
    except:
        continue

dfObj1.sort_values(by=['AIC'])

# Fit the ARIMA model
model = ARIMA(train, order=(2, 1, 3))
results_Arima = model.fit()
print(results_Arima.summary())

# Predict on the Test Set using this model and evaluate the model on the test set using RMSE and MAPE
pred_start = test.index[0]
pred_end = test.index[-1]
ARIMA_predictions = results_Arima.predict(start=pred_start, end=pred_end)
ARIMA_pred = ARIMA_predictions.cumsum()

plt.plot(train, label='Training Data')
plt.plot(test, label='Test Data')
plt.plot(test.index, ARIMA_predictions, label='Predicted Data - ARIMA')
plt.legend(loc='best')
plt.grid()

ARIMA_predictions_df = pd.DataFrame(ARIMA_predictions)
residuals = test['CO2 ppm'] - ARIMA_predictions_df['predicted_mean']
qqplot(residuals, line="s")

from sklearn.metrics import mean_squared_error
rmse = mean_squared_error(test['CO2 ppm'], ARIMA_predictions_df['predicted_mean'], squared=False)
print(rmse)

def MAPE(y_true, y_pred):
    return np.mean((np.abs(y_true - y_pred)) / y_true) * 100

mape = MAPE(test['CO2 ppm'].values, ARIMA_predictions_df['predicted_mean'].values)
print(mape)

# SARIMA Model
p = q = range(0, 3)
d = range(1, 2)
pdq = list(itertools.product(p, d, q))
model_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
print('Examples of parameter combinations for Model...')
print(f'Model: {pdq[1]}{model_pdq[1]}')
print(f'Model: {pdq[1]}{model_pdq[2]}')
print(f'Model: {pdq[2]}{model_pdq[3]}')
print(f'Model: {pdq[2]}{model_pdq[4]}')

dfObj2 = pd.DataFrame(columns=['param', 'seasonal', 'AIC'])

for param in pdq:
    for param_seasonal in model_pdq:
        mod = SARIMAX(train, order=param, seasonal_order=param_seasonal, enforce_stationarity=False, enforce_invertibility=False)
        results_SARIMA = mod.fit()
        print(f'SARIMA{param}x{param_seasonal}12 - AIC:{results_SARIMA.aic}')
        dfObj2 = dfObj2.append({'param': param, 'seasonal': param_seasonal, 'AIC': results_SARIMA.aic}, ignore_index=True)

dfObj2.sort_values(by=['AIC'])

model = SARIMAX(train, order=(1, 1, 0), seasonal_order=(1, 1, 2, 12))
model_Sarima = model.fit()
print(model_Sarima.summary())

SARIMA_predictions = model_Sarima.predict(start=pred_start, end=pred_end)

plt.plot(train, label='Training Data')
plt.plot(test, label='Test Data')
plt.plot(test.index, SARIMA_predictions, label='Predicted Data - SARIMA')
plt.legend(loc='best')
plt.grid()

rmse = mean_squared_error(test['CO2 ppm'], SARIMA_predictions, squared=False)
print(rmse)

mape = MAPE(test['CO2 ppm'], SARIMA_predictions)
print(mape)

model_Sarima.plot_diagnostics(figsize=(16, 8))
plt.show()

# Fitting model on whole data
model = SARIMAX(df, order=(1, 1, 0), seasonal_order=(1, 1, 2, 12), enforce_stationarity=False, enforce_invertibility=False)
model_Sarima = model.fit()
print(model_Sarima.summary())

# Forecast with confidence interval
forecast = model_Sarima.forecast(steps=24)
pred95 = model_Sarima.get_forecast(steps=24).conf_int()

axis = df.plot(label='Observed', figsize=(15, 8))
forecast.plot(ax=axis, label='Forecast', alpha=0.7)
axis.fill_between(forecast.index, pred95['lower CO2 ppm'], pred95['upper CO2 ppm'], color='k', alpha=.15)
axis.set_xlabel('Year-Months')
axis.set_ylabel('CO2 ppm')
plt.legend(loc='best')
plt.show()

# END

