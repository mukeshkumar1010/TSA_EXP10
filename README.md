# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL
## Date: 10.05.25

## AIM:
To implement SARIMA model using python.
## ALGORITHM:
1. Explore the dataset
2. Check for stationarity of time series
3. Determine SARIMA models parameters p, q
4. Fit the SARIMA model
5. Make time series predictions and Auto-fit the SARIMA model
6. Evaluate model predictions
## PROGRAM:
### Name : Mukesh Kumar S
### Register Number : 212223240099
```
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv('AirPassengers.csv')
data.head()
data.set_index('Month', inplace=True)
time_series = data['#Passengers']
plt.figure(figsize=(10, 6))
plt.plot(time_series)
plt.title('Passengers Time Series Plot')
plt.xlabel('Month')
plt.ylabel('Passengers')
plt.show()
def test_stationarity(timeseries):
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    for key, value in result[4].items():
        print(f'Critical Value {key}: {value}')
    if result[1] < 0.05:
        print("The time series is stationary.")
    else:
        print("The time series is not stationary.")

test_stationarity(time_series)
time_series_diff = time_series.diff().dropna()
print("\nAfter Differencing:")
test_stationarity(time_series_diff)

p, d, q = 1, 1, 1
P, D, Q, m = 1, 1, 1, 12 
model = SARIMAX(time_series, order=(p, d, q), seasonal_order=(P, D, Q, m), enforce_stationarity=False, enforce_invertibility=False)
sarima_fit = model.fit(disp=False)
print(sarima_fit.summary())
forecast_steps = 12 
forecast = sarima_fit.get_forecast(steps=forecast_steps)
forecast_ci = forecast.conf_int()

time_series.index = pd.to_datetime(time_series.index).tz_localize(None)
forecast.predicted_mean.index = pd.to_datetime(forecast.predicted_mean.index).tz_localize(None)
forecast_ci.index = pd.to_datetime(forecast_ci.index).tz_localize(None)


plt.figure(figsize=(10, 6))
plt.plot(time_series, label='Historical Data')
plt.plot(forecast.predicted_mean, label='Forecast', color='red')
plt.fill_between(forecast_ci.index,
                 forecast_ci.iloc[:, 0],
                 forecast_ci.iloc[:, 1], color='pink', alpha=0.3)
plt.title('SARIMA Forecast of Passengers')
plt.xlabel('Month')
plt.ylabel('Passengers')
plt.legend()
plt.show()

from sklearn.metrics import mean_absolute_error
test_data = time_series[-forecast_steps:]
pred_data = forecast.predicted_mean[:len(test_data)]
mae = mean_absolute_error(test_data, pred_data)
print('Mean Absolute Error:', mae)
```

### OUTPUT:

#### Time Series Plot:

![Output 1](https://github.com/user-attachments/assets/d9bed813-2eb1-4e32-862f-01f92257aa1e)

#### After Differencing:

![difference](https://github.com/user-attachments/assets/44ef9e06-899e-4a43-833e-d8e19ed216e5)

#### SARIMA Forecast:

![Output2](https://github.com/user-attachments/assets/90b87940-2ef5-4b17-b6d4-87bd3360c2a4)

#### Mean Absolute Error:

![mean](https://github.com/user-attachments/assets/ed8f4bf1-1ed8-4d2c-b7a0-f1617f8031fb)

### RESULT:
Thus the program run successfully based on the SARIMA model.
