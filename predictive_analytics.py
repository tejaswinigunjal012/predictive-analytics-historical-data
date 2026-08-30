# Predictive Analytics Using Historical Data
# Thiranex Student Task

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.datasets import get_rdataset
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error


# 1. Load the historical data
# AirPassengers contains monthly international airline passenger totals.
# The 144 historical observations are included below, so the project can run offline.
passenger_values = [
112,118,132,129,121,135,148,148,136,119,104,118,
115,126,141,135,125,149,170,170,158,133,114,140,
145,150,178,163,172,178,199,199,184,162,146,166,
171,180,193,181,183,218,230,242,209,191,172,194,
196,196,236,235,229,243,264,272,237,211,180,201,
204,188,235,227,234,264,302,293,259,229,203,229,
242,233,267,269,270,315,364,347,312,274,237,278,
284,277,317,313,318,374,413,405,355,306,271,306,
315,301,356,348,355,422,465,467,404,347,305,336,
340,318,362,348,363,435,491,505,404,359,310,337,
360,342,406,396,420,472,548,559,463,407,362,405,
417,391,419,461,472,535,622,606,508,461,390,432
]

data = pd.DataFrame({
    "time": pd.date_range(start="1949-01-01", periods=len(passenger_values), freq="MS"),
    "passengers": passenger_values
}).set_index("time")

print("First five records:")
print(data.head())
print("\nDataset shape:", data.shape)
print("\nMissing values:", data["passengers"].isna().sum())


# 2. Basic preprocessing
data["passengers"] = pd.to_numeric(data["passengers"], errors="coerce")
data = data.dropna()

# Make sure the monthly frequency is retained.
data = data.asfreq("MS")


# 3. Visualize the historical trend
plt.figure(figsize=(10, 5))
plt.plot(data.index, data["passengers"], linewidth=2)
plt.title("Monthly Airline Passengers (1949–1960)")
plt.xlabel("Year")
plt.ylabel("Number of Passengers (thousands)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("historical_trend.png", dpi=160)
plt.show()


# 4. Train-test split
# The last 12 months are kept for testing.
train = data.iloc[:-12]
test = data.iloc[-12:]

print("\nTraining records:", len(train))
print("Testing records:", len(test))


# 5. Build the ARIMA model
# ARIMA(2,1,2) is used as a simple time-series forecasting model.
model = ARIMA(train["passengers"], order=(2, 1, 2))
model_fit = model.fit()

print("\nModel summary:")
print(model_fit.summary())


# 6. Predict the test period
pred = model_fit.forecast(steps=len(test))
pred.index = test.index

mae = mean_absolute_error(test["passengers"], pred)
mse = mean_squared_error(test["passengers"], pred)
rmse = np.sqrt(mse)

print("\nEvaluation results")
print("MAE :", round(mae, 2))
print("MSE :", round(mse, 2))
print("RMSE:", round(rmse, 2))


# 7. Plot actual vs predicted values
plt.figure(figsize=(10, 5))
plt.plot(train.index, train["passengers"], label="Training data")
plt.plot(test.index, test["passengers"], label="Actual test data", linewidth=2)
plt.plot(pred.index, pred, label="Predicted", linestyle="--", linewidth=2)
plt.title("Actual vs Predicted Airline Passengers")
plt.xlabel("Year")
plt.ylabel("Passengers (thousands)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=160)
plt.show()


# 8. Forecast the next 12 months using the complete historical data
final_model = ARIMA(data["passengers"], order=(2, 1, 2)).fit()

future = final_model.forecast(steps=12)
future_dates = pd.date_range(
    start=data.index[-1] + pd.offsets.MonthBegin(1),
    periods=12,
    freq="MS"
)
future.index = future_dates

forecast_table = pd.DataFrame({
    "Month": future.index.strftime("%Y-%m"),
    "Forecasted Passengers (thousands)": np.round(future.values, 2)
})

print("\nNext 12-month forecast:")
print(forecast_table.to_string(index=False))

forecast_table.to_csv("forecast_12_months.csv", index=False)


# 9. Plot the future forecast
plt.figure(figsize=(11, 5))
plt.plot(data.index, data["passengers"], label="Historical data", linewidth=2)
plt.plot(future.index, future.values, label="12-month forecast",
         linestyle="--", marker="o", linewidth=2)
plt.title("Airline Passenger Forecast for the Next 12 Months")
plt.xlabel("Year")
plt.ylabel("Passengers (thousands)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("future_forecast.png", dpi=160)
plt.show()


print("\nProject completed successfully.")
