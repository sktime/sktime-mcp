# Persistence and Export

`sktime-mcp` allows you to save your work for future sessions and export your analysis as production-ready code by working with your AI assistant.

## Saving and Loading Your Models

If you've trained a model that you want to use later, you can ask your assistant to save it to your local filesystem.

### Saving a Model
Tell your assistant where you want to store the model.

> *Example: "Save this fitted ARIMA model to /home/user/models/my_model."*

The assistant will use the `save_model` tool to persist the estimator using MLflow. This ensures that the model's state (including its trained weights) is preserved.

### Loading a Model
In a new session, you can bring back a previously saved model by telling the assistant where it is.

> *Example: "Load my saved model from /home/user/models/my_model."*

Once loaded, you can immediately start forecasting or evaluating the model as if you had just trained it.

## Exporting Your Workflow as Code

One of the most powerful features of `sktime-mcp` is the ability to turn a conversation into a standalone Python script. This is perfect for moving from exploration to production.

### Generating Python Code
Simply ask your assistant to provide the code for your current workflow.

> *Example: "Give me the Python code for the model we just built."*

The assistant will generate a script that includes:
- All necessary **library imports** (`sktime`, `pandas`, etc.).
- The exact **model configuration** and hyperparameter settings.
- The **pipeline structure** (if you used multiple steps).
- An **example workflow** showing how to fit and predict with the model.

This allows you to reproduce your results exactly, even without the MCP server running.

## Important Considerations

- **Absolute Paths**: When saving or loading, always provide absolute paths to ensure the assistant can find the correct location on your system.
- **Environment Matching**: When running exported code, make sure your Python environment has the same libraries installed as the server (e.g., `sktime`).
