
# sktime-mcp

**[Read the Official Documentation](http://sktime.github.io/sktime-mcp/)** | **[PyPI Package](https://pypi.org/project/sktime-mcp/)**

<div class="hero-text">
  <h1>The Semantic Engine for Time-Series</h1>
  <p style="font-size: 1.2rem; margin-bottom: 2rem;">
    Enables Large Language Models to discover, reason about, and execute 
    sktime's advanced forecasting algorithms on real-world data.
  </p>
</div>

> **Why sktime-mcp?**
> Combines **LLM reasoning** with **time-series precision**. 
> Instead of hallucinating Python code, your agent interacts with a strictly typed, 
> safe, and stateful API to perform complex forecasting tasks.

---

## 👋 Who is this for?

sktime‑mcp is designed for:

- **Developers** building LLM agents that need reliable, production‑grade time‑series forecasting.
- **Data scientists** who want to expose sktime workflows to language models without unsafe code generation.
- **Platform teams** integrating forecasting capabilities into tools like Claude Desktop, Cursor, or custom MCP clients.

If you are new to MCP‑based workflows, start with the **Quick Start** below, then explore the **Use Cases** and **User Guide** for deeper examples.

---

## 🔥 Key Features

- **Semantic Discovery:** Find the perfect estimator using semantic similarity and capability tags (e.g., "probabilistic forecaster that handles missing data").
- **Safe Composition:** Build complex pipelines (Transformer → Forecaster) with built-in validation to ensure components are compatible before execution.
- **Universal Data Loading:** Seamlessly ingest data from SQL, Pandas, Parquet, Excel, and CSV files.
- **Execution Runtime:** Stateful engine that manages object lifecycles, fitting, and predicting, all via simple JSON-RPC tools.

---


## ⚡ Quick Start

Get up and running in seconds. Use with **Claude Desktop**, **Cursor**, or any MCP-compatible client.

### 1. Install

**Zero-install via uvx (recommended):** if you have [uv](https://github.com/astral-sh/uv) installed, skip this step — uvx fetches and runs the package automatically when your MCP client starts.

```bash
# Or install explicitly with pip
pip install sktime-mcp
```

When contributing, install from source:

```bash
git clone https://github.com/sktime/sktime-mcp.git
cd sktime-mcp
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Connect (Claude Desktop / Claude Code Config)
Add this to your `claude_desktop_config.json`:

**With uvx (no prior install needed):**
```json
{
  "mcpServers": {
    "sktime": {
      "command": "uvx",
      "args": ["sktime-mcp"]
    }
  }
}
```

**With pip-installed package:**
```json
{
  "mcpServers": {
    "sktime": {
      "command": "sktime-mcp"
    }
  }
}
```

---


## 📚 Documentation Map

| Section | Description |
| :--- | :--- |
| [**Quickstart**](quickstart.md) | Fastest route to running the sktime-mcp server and connecting it to a client. |
| [**User Guide**](user-guide.md) | Comprehensive manual on using tools, workflows, and best practices. |
| [**Concepts**](concepts.md) | Core concepts, handles, and asynchronous operations. |
| [**Architecture**](developer/architecture.md) | High-level system design, data flow, and limitations. |
| [**Contributing**](developer/contributing.md) | Contributing guidelines, testing, and extending the server. |

---


## 🚀 Get Started

- See [Quickstart](quickstart.md) to start the server.
- See [User Guide](user-guide.md) for detailed instructions and advanced features.

[Get Started Now](quickstart.md){ .md-button .md-button--primary }
