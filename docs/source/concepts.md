# Core Concepts

`sktime-mcp` is a specialized interface that allows you to perform advanced time-series analysis by collaborating with an AI assistant. It bridges the gap between natural language requests and the rigorous execution environment of the `sktime` library.

## Collaborative Model Discovery

Instead of requiring you to know the exact names of hundreds of forecasting models, `sktime-mcp` enables a discovery-based workflow. You can ask your assistant to find models based on your data's characteristics:
- **"Find models for multivariate data"**
- **"Which estimators handle missing values?"**
- **"Show me probabilistic forecasters."**

The system queries the `sktime` registry in real-time, ensuring your assistant always has access to the most up-to-date models and their metadata.

## Stateful Interaction (Handles)

To allow for complex, multi-turn conversations, the server maintains a stateful runtime. This is managed through **Handles**.

### How it works for you
When you ask the assistant to load a dataset or create a model, the server creates that object in memory and assigns it a "handle" (a unique ID).
- You don't need to track these IDs yourself.
- You can refer to objects naturally: *"Use the model we just created"* or *"Run that forecast on the sales data I loaded earlier."*
- The assistant manages the handles behind the scenes to ensure your requests are executed on the correct objects.

### Memory Management
Because these objects stay in memory to support follow-up questions, you can tell the assistant to "clear the session" or "release the data" when you are finished to free up system resources.

## Asynchronous Background Jobs

Time-series forecasting can sometimes be computationally intensive. If a task (like training a deep learning model) will take a long time, you can ask the assistant to **run it in the background**.

- This allows you to continue the conversation or perform other tasks while the model trains.
- You can ask for a status update at any time: *"Is the model training finished yet?"*
- The assistant will notify you once the results are ready.

## Safety and Reproducibility

### Secure Execution
Unlike systems that generate and run arbitrary code (which can be prone to errors or security risks), `sktime-mcp` uses a strictly defined set of tools. Your assistant interacts with `sktime` through a validated API, ensuring that operations are safe and predictable.

### From Conversation to Code
Once you've found a workflow that works, you can turn your conversation into a permanent asset. Ask the assistant to **"export the Python code,"** and it will generate a standalone script that reproduces your entire analysis exactly.
