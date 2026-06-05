# Pipelines and Composition

`sktime-mcp` allows you to build complex, multi-step workflows that combine data transformation with forecasting by simply describing them to your AI assistant.

## Describing a Pipeline

Instead of writing complex code to chain models together, you can describe the sequence of steps you want to perform.

> *Example: "I want to deseasonalize and detrend my data before applying an ARIMA model."*

Your assistant will understand these steps and use the `sktime` pipeline architecture to connect them correctly.

## Validation

Before a pipeline is created, your assistant can verify that the steps you've described are scientifically compatible.

> *Example: "Check if combining a Detrender with an ARIMA model is valid."*

The assistant will use the `validate_pipeline` tool to ensure that:
- Transformers are placed before the final forecaster.
- Data types (like univariate or multivariate) match across all steps.

## Creating and Using Pipelines

Once you are satisfied with the design, tell the assistant to build the pipeline.

> *Example: "Create a pipeline with those steps and use it to forecast the airline dataset."*

A pipeline is treated just like a single model. You can ask to fit it, evaluate it, or export its code. The system automatically handles the flow of data through every stage of the pipeline on your behalf.

## Common Use Cases

You can ask your assistant to help with:
- **Detrending**: Removing long-term trends from your data.
- **Deseasonalization**: Removing periodic patterns (e.g., monthly seasonality).
- **Imputation**: Filling missing values automatically as part of the model.
- **Scaling**: Standardizing data ranges for better model performance.
