
# sktime-mcp

**[Read the Official Documentation](http://sktime.github.io/sktime-mcp/)** | **[PyPI Package](https://pypi.org/project/sktime-mcp/0.1.0/)**

<div class="hero-text">
  <h1>The Semantic Engine for Time-Series</h1>
  <p style="font-size: 1.2rem; margin-bottom: 2rem;">
    Empower Large Language Models to discover, reason about, and execute 
    sktime's advanced forecasting algorithms on real-world data.
  </p>
</div>

> **Why sktime-mcp?**
> Bridging the gap between **LLM reasoning** and **time-series precision**. 
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

```bash
# Install directly from PyPI (https://pypi.org/project/sktime-mcp/0.1.0/)
pip install sktime-mcp
```

Alternatively, when contributing, use GitHub to install from source:

```bash
git clone https://github.com/sktime/sktime-mcp.git
cd sktime-mcp
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Run
```bash
sktime-mcp
```

### 3. Connect (Claude Desktop Config)
Add this to your `claude_desktop_config.json`:
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
| [**Use Cases**](use-cases.md) | Step-by-step workflows for coders and business users. |
| [**User Guide**](user-guide.md) | Comprehensive manual on using tools, workflows, and best practices. |
| [**Usage Examples**](usage-examples.md) | Example scripts and advanced usage patterns. |
| [**Data Sources**](data-sources.md) | Deep dive into loading data from SQL, Files, and Pandas. |
| [**Architecture**](architecture.md) | High-level system design, data flow, and limitations. |
| [**Implementation**](implementation.md) | Detailed code walkthrough and file breakdown. |
| [**Developer Guide**](dev-guide.md) | Contributing guidelines, testing, and extending the server. |

---


## 🚀 Get Started

- See [Use Cases](use-cases.md) for step-by-step workflows.
- See [User Guide](user-guide.md) for detailed instructions and advanced features.

[Get Started Now](use-cases.md){ .md-button .md-button--primary }
