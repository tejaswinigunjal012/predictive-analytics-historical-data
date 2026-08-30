# Predictive Analytics Using Historical Data

## Objective
The aim of this project is to use historical data to understand a trend and forecast future values.

## Dataset
AirPassengers — monthly international airline passenger totals from 1949 to 1960.

## Model
ARIMA (2,1,2) time-series model.

## Steps
1. Load the historical dataset.
2. Check and clean the data.
3. Convert the time column into a monthly date index.
4. Visualize the historical trend.
5. Split the data into training and testing sets.
6. Train an ARIMA model.
7. Evaluate predictions using MAE, MSE and RMSE.
8. Forecast the next 12 months.
9. Visualize the forecast.

## Requirements
Install the libraries with:

pip install pandas numpy matplotlib statsmodels scikit-learn

Then run:

python predictive_analytics.py

The program creates:
- historical_trend.png
- actual_vs_predicted.png
- future_forecast.png
- forecast_12_months.csv
