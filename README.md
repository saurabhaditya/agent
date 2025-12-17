# Review Genie - LLM Agent with ReAct Framework

An AI-powered agent that combines Amazon product search (via FAISS vector store) with real-time web search (via Tavily) using LangChain's ReAct framework.

## Features

- **Amazon Product Search**: Semantic search over product data using FAISS vector store
- **Web Search**: Real-time information retrieval via Tavily Search API
- **Conversation Memory**: Session-based chat history with ConversationSummaryMemory
- **ReAct Reasoning**: Chain-of-thought reasoning for intelligent tool selection
- **Gradio UI**: Web interface for interactive conversations
- **Test Suite**: Automated tests to verify agent functionality

## Prerequisites

- macOS/Linux
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda
- OpenAI API key
- Tavily API key

## Installation

### 1. Install Conda (if not installed)

```bash
# macOS with Homebrew
brew install --cask miniconda

# Initialize conda in your shell (only needed once)
conda init zsh  # or bash
source ~/.zshrc  # or open a new terminal
```

### 2. Create Conda Environment

```bash
# Create environment with Python 3.11 (required for LangChain compatibility)
conda create -n agent_env python=3.11 -y

# Activate the environment
conda activate agent_env
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Environment Variables

Add to your `~/.bash_profile` or `~/.zshrc`:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export TAVILY_API_KEY="your-tavily-api-key"
```

Then source it:
```bash
source ~/.bash_profile  # or ~/.zshrc
```

## Usage

### Run Gradio Web Interface (default)

```bash
conda activate agent_env
python agent.py
# or explicitly:
python agent.py --gradio
```

Opens a web UI at `http://localhost:7860` with a shareable public link.

### Run Automated Tests

```bash
conda activate agent_env
python agent.py --test
```

Runs 4 tests:
1. Amazon Product Search (vector store)
2. Product Reviews Query (vector store)
3. Web Search (Tavily)
4. General Knowledge Query (web search)

## Project Structure

```
agent/
├── agent.py           # Main agent implementation
├── faiss_index/       # FAISS vector store
│   ├── index.faiss    # Vector embeddings
│   └── index.pkl      # Metadata
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

## Dependencies & Gotchas

### Python Version Requirement

**Python >= 3.10 is required.** The `langchain-community` package does not support Python 3.9 or earlier.

```bash
# Check your Python version
python --version

# If using system Python < 3.10, you MUST use conda:
conda create -n agent_env python=3.11 -y
```

### LangChain API Migration (v1.x → v2.x)

LangChain has undergone significant API changes. The following imports have changed:

| Old Import | New Import |
|------------|------------|
| `from langchain.tools.retriever import ...` | `from langchain_core.tools.retriever import ...` |
| `from langchain import hub` | `from langsmith import Client` then `client.pull_prompt()` |
| `from langchain.agents import AgentExecutor` | `from langchain_classic.agents import AgentExecutor` |
| `from langchain.memory import ConversationSummaryMemory` | `from langchain_classic.memory import ConversationSummaryMemory` |

### Required Packages

Key packages and their purposes:

| Package | Purpose |
|---------|---------|
| `langchain` | Core LangChain framework |
| `langchain-openai` | OpenAI LLM integration |
| `langchain-community` | Community tools (FAISS, Tavily) |
| `langchain-core` | Core abstractions and tools |
| `langchainhub` | Prompt hub access |
| `langsmith` | LangSmith client for pulling prompts |
| `faiss-cpu` | Vector similarity search |
| `tavily-python` | Tavily search API |
| `gradio` | Web UI framework |

### Deprecation Warnings

You may see deprecation warnings for:

1. **TavilySearchResults**: Will move to `langchain-tavily` package
   ```bash
   pip install -U langchain-tavily
   ```

2. **ConversationSummaryMemory**: Legacy memory system
   - Still functional, will be updated in future LangChain versions

These warnings don't affect functionality.

### FAISS Vector Store

The vector store must be loaded with `allow_dangerous_deserialization=True` because it uses pickle:

```python
vector = FAISS.load_local(
    "./faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)
```

**Security Note**: Only load FAISS indexes from trusted sources.

### Conda Activation

If `conda activate` doesn't work after installation:

```bash
# Initialize conda in your shell
conda init zsh  # or bash

# Then either:
source ~/.zshrc  # Source the updated config
# OR
# Open a new terminal window
```

## API Keys

### OpenAI API Key
- Get from: https://platform.openai.com/api-keys
- Used for: GPT-4o-mini LLM and text embeddings

### Tavily API Key
- Get from: https://tavily.com/
- Used for: Real-time web search
- Free tier available

## How It Works

1. **Query Processing**: User input is processed by the ReAct agent
2. **Tool Selection**: Agent decides whether to use:
   - `amazon_product_search`: For product-related queries
   - `tavily_search_results_json`: For general/current information
3. **Reasoning Loop**: Agent may chain multiple tool calls
4. **Response Generation**: Final answer synthesized from tool outputs

## Troubleshooting

### "ModuleNotFoundError: No module named 'langchain_classic'"

```bash
pip install langchain  # Includes langchain_classic
```

### "OPENAI_API_KEY environment variable not set"

```bash
export OPENAI_API_KEY="sk-..."
# Or add to ~/.bash_profile and source it
```

### "Vector store not found"

Ensure `faiss_index/` directory exists with `index.faiss` and `index.pkl` files.

### Tests fail with empty responses

- Check API keys are valid and have credits
- Verify internet connection for Tavily searches

## License

MIT
