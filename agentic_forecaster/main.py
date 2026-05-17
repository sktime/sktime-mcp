import pandas as pd
from sktime.forecasting.base import ForecastingHorizon
import os
from agent import AgenticForecaster

script_dir = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(script_dir, "data", "train.csv")
df = pd.read_csv(train_path)
df["date"] = pd.to_datetime(df["date"])
df = df[(df["store_nbr"] == 1) & (df["family"] == "GROCERY I")].copy()
df = df.groupby("date")["sales"].sum().sort_index().tail(90)
y = pd.Series(df)

fh = ForecastingHorizon(
    pd.date_range(start=y.index[-1] + pd.Timedelta(days=1), periods=30, freq="D"),
    is_relative=False
)

prompt = "Forecast grocery sales for the next 30 days. I need prediction intervals for uncertainty."
model = AgenticForecaster(prompt=prompt)
print(f"Agentic Prompt: {prompt}")
model.fit(y)
print(f"Agent Logic: {model.explain()}")
predictions = model.predict(fh)
print("\nNext 30 Days Forecast:\n")
print(predictions)
