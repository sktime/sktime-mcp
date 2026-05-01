# Quickstart

This guide provides the fastest route to running the `sktime-mcp` server and connecting it to a client.

## 1. Start the Server

Once installed, you can start the MCP server directly from your terminal:

```bash
sktime-mcp
```

By default, the server communicates over standard input/output (stdio).

## 2. Connect to a Client

To use `sktime-mcp` with an AI assistant, you must configure the client to launch the server.

### Claude Desktop

Add the following configuration to your `claude_desktop_config.json`:

**Linux**: `~/.config/Claude/claude_desktop_config.json`  
**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`  
**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "sktime": {
      "command": "sktime-mcp"
    }
  }
}
```

### Cursor or VS Code (MCP Extensions)

Configure the command `sktime-mcp` in the MCP settings section of your IDE or extension.

## 3. Verify Functionality

Once connected, you can verify the server is working by asking your assistant:

> "What forecasting models are available in sktime?"

The assistant will use the `list_estimators` tool and return a list of available models for your review.

## 4. Run Your First Forecast

Try an end-to-end workflow by directing your assistant to use a built-in dataset:

> "Load the 'airline' demo dataset and forecast the next 12 months using a NaiveForecaster."

The assistant will handle the technical steps — locating the data, instantiating the model, and fitting it — before presenting the final predictions to you in the chat.

## Next Steps

- Explore **Core Concepts** to understand how the system manages state.
- Refer to the **User Guide** for instructions on loading your own data and building pipelines.
