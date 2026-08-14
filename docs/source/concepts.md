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

Time-series work can be computationally intensive. If a task (like training a deep learning model) will take a long time, you can ask the assistant to **run it in the background**.

Under the hood this is a single flag — `run_async: true` — accepted by the four
tools that can be slow: `fit`, `predict`, `evaluate`, and `load_data_source`.
There are no separate "async" tools. The call returns a **job ID** straight away
instead of a result.

- This allows you to continue the conversation while the model trains.
- You can ask for a status update at any time: *"Is the model training finished yet?"* — the assistant checks the job for you.
- You can also cancel work that is taking too long: *"Stop that training run."*

```{note}
Nothing arrives unprompted. The server has no way to push a message to you when a
job finishes, so the assistant only learns the job is done when it checks. Ask if
you want to know.
```

See the {doc}`tool-reference` for the job tools and the exact flag.

## Safety and Reproducibility

### Execution Model
For modelling, `sktime-mcp` does not ask an LLM to write Python and then run it. Your assistant selects from a fixed set of tools with validated inputs, and models are constructed through sktime's own registry — so a typo or a hallucinated class name fails cleanly instead of executing something unintended.

```{warning}
One tool is deliberately outside that boundary. `run_command` runs an arbitrary
shell command with your account's privileges, so the assistant can install a
missing optional dependency mid-session. It is powerful and it is **not**
sandboxed.

Most MCP clients let you approve or deny tools individually. Keep `run_command`
on ask-every-time, or turn it off, unless you actively want the assistant
installing packages and inspecting files for you. See {doc}`tool-reference`.
```

The server has no authentication layer of its own: it runs as you, with your
filesystem permissions. Treat it as a local trusted tool, and use a container if
you want a harder boundary.

### From Conversation to Code
Once you've found a workflow that works, you can turn your conversation into a permanent asset. Ask the assistant to **"export the Python code,"** and it will generate a standalone script that reproduces your entire analysis exactly.
