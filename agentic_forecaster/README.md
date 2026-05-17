# Agentic Forecaster for sktime-mcp

This module implements an **Agentic Forecaster** that leverages the `sktime-mcp` Model Context Protocol (MCP) server to perform intelligent model selection and forecasting based on natural language prompts.

## Overview

Traditional forecasting workflows require manual model selection based on data characteristics. The `AgenticForecaster` automates this by:
1. **Semantic Reasoning**: Analyzing the user's prompt (e.g., "seasonal data", "fast execution") against the `sktime` registry's capability tags.
2. **Dynamic Tool Use**: Interacting with the MCP server to instantiate models, load data, and execute forecasts.
3. **Exogenous Support**: Automatically handling covariates (X) when provided, enabling professional-grade forecasting on complex datasets.

## Key Components

- **`agent.py`**: The core `AgenticForecaster` class. It manages the registry-driven reasoning and the interface with the MCP execution engine.
- **`main.py`**: A demonstration script showcasing the agent's ability to forecast retail sales data (Corporación Favorita dataset) using a simple English prompt.

## Usage Example

```python
from agentic_forecaster.agent import AgenticForecaster

# Initialize the agent
agent = AgenticForecaster()

# Execute a forecast with a natural language requirement
result = agent.fit_predict(
    prompt="I need a model that handles seasonality and is fast to train.",
    dataset="favorita_subset",  # Or use a data_handle
    horizon=30
)

print(f"Selected Model: {result['selected_model']}")
print(f"Explanation: {result['explanation']}")
```

## Contributions to sktime-mcp

This agentic workflow drove several core improvements to the `sktime-mcp` project:
- **Exogenous Support**: Added `exog_handle` support to the `fit_predict` tool stack to enable covariates in agentic workflows.
- **Evaluation Logic**: Fixed cross-validation fold calculation bugs in the `evaluate` tool to ensure agents receive accurate performance metrics.
- **Registry Visibility**: Improved docstring handling to ensure the agent can read full model descriptions for better decision making.
