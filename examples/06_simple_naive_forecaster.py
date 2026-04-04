from sktime.datasets import load_airline
from sktime.forecasting.naive._naive import NaiveForecaster

forecaster = NaiveForecaster()

# Example usage:
# Load data

y = load_airline()

# Fit the model
forecaster.fit(y)

# Make predictions
fh = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # 12-step ahead forecast
predictions = forecaster.predict(fh=fh)
print(predictions)
