from sktime.datasets import load_airline
from sktime.forecasting.compose import TransformedTargetForecaster
from sktime.forecasting.naive._naive import NaiveForecaster
from sktime.transformations.compose import TransformerPipeline
from sktime.transformations.series.detrend._deseasonalize import Deseasonalizer
from sktime.transformations.series.detrend._detrend import Detrender

# Define the components
step_0 = Deseasonalizer()
step_1 = Detrender()
step_2 = NaiveForecaster()

# Build the pipeline
transformer_chain = TransformerPipeline([("step_0", step_0), ("step_1", step_1)])
forecaster = TransformedTargetForecaster(
    [
        ("transformers", transformer_chain),
        ("forecaster", step_2),
    ]
)

# Example usage:
# Load data

y = load_airline()

# Fit the model
forecaster.fit(y)

# Make predictions
fh = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # 12-step ahead forecast
predictions = forecaster.predict(fh=fh)
print(predictions)
