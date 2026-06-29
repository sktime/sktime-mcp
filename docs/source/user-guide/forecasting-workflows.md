# Forecasting Workflows

This guide covers how to discover models, train them on your data, and generate forecasts by collaborating with your AI assistant.

## Model Discovery

With hundreds of estimators available in `sktime`, you can ask your assistant to find the perfect model for your task.

### Finding the Right Model
Ask your assistant to search for models based on their capabilities or specific tasks.

> *Example: "Find models that can handle multivariate data" or "Show me probabilistic forecasters for forecasting."*

The assistant will use the `query_registry` tool to find matches. Common capabilities you might ask for include:
- **Prediction Intervals**: For probabilistic forecasting.
- **Multivariate Support**: For data with multiple related time series.
- **Missing Data Handling**: For datasets with gaps.

### Inspecting a Model
If you want to know more about a specific model, ask for its details.

> *Example: "Tell me more about the ARIMA model and its parameters."*

The assistant will provide a description and a list of available hyperparameters.

## The Fit-Predict Workflow

Once you have a model and data, you can ask the assistant to perform the forecast.

### Running a Forecast
You can request a forecast using either a demo dataset or your own loaded data.

> *Example: "Forecast the next 12 months using ARIMA on the airline dataset."*

The assistant will handle the instantiation, fitting, and prediction steps in one go. If you want to customize the model, you can specify parameters:

> *Example: "Set up an ARIMA model with order (1, 1, 1) and forecast 6 steps ahead on my loaded data."*

## Evaluation

To see how well a model performs, ask your assistant to evaluate it.

> *Example: "Evaluate this model using 3-fold cross-validation on the airline dataset."*

The assistant will run the evaluation and report back with performance metrics like Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

## Best Practices for Users

1.  **Use Natural Language**: You don't need to know the technical tool names. Just tell the assistant what you want to achieve (e.g., "I want to forecast sales for next year").
2.  **Request Reproducibility**: After a successful experiment, ask for the code: "Give me the Python code for this workflow." This allows you to run the same analysis independently later.
3.  **Manage Your Session**: If you have been working with many different models, you can tell the assistant to "release all handles" or "clear the session" to free up server memory.
4.  **Leverage Background Jobs**: For very large datasets, tell the assistant to "run this in the background" so you can continue the conversation while the model trains.
