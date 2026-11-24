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
The assistant calls named MCP tools with advertised input schemas and tool-specific
validation. `query_registry` reads the installed sktime registry;
`describe_component` resolves registered component names without evaluating input
or importing a client-supplied module. Descriptions include dependency availability,
missing requirements and an installation hint when applicable.

`instantiate` has a different trust boundary: it passes its specification to
sktime's `craft()`, which evaluates Python expressions. A registry-backed interface
does not make that expression evaluator a sandbox. Only use specifications from
trusted clients and review them before execution.

```{warning}
`run_command` also executes arbitrary shell commands with the server account's
privileges. Keep it on ask-every-time in the client, or disable it unless you need
that capability. Neither a tool schema nor HTTP authentication restricts what an
authorized call can do on the host.
```

### Local and HTTP Access
Local stdio runs as the user who starts the process and needs no HTTP token. The
HTTP/SSE transport requires `SKTIME_MCP_HTTP_TOKEN` and an
`Authorization: Bearer <token>` header on protected routes. Only the `/` health
check is public. HTTP startup without a token is refused unless
`SKTIME_MCP_HTTP_INSECURE=true` is explicitly enabled for local testing.

Host/origin allowlists protect the HTTP transport as described in the README.
Authentication controls who can call the server; it does not provide filesystem
isolation. A container can reduce access when configured with limited mounts and
permissions. See {doc}`developer/architecture` for the full boundary.

### From Conversation to Code
`export_code` reconstructs an estimator or pipeline from its stored specification,
or emits a loader for a saved model. `include_fit_example=true` adds an example for
a built-in dataset. It does not record the whole conversation, custom data loading,
cleaning decisions, fitted weights or the Python environment.

Use `save_data` for data and predictions, `save_model` for a fitted estimator, and
record package versions plus the steps needed to reproduce your analysis. Review
and run exported code in the environment where you intend to use it.
