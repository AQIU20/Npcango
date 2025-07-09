# NPCango - Intelligent NPC System

An AI-powered NPC system built with Agno 0.15.x that enables interactive blacksmith NPCs in games. The system features long-term memory, knowledge retrieval, and inventory management capabilities.

## Features

- 🧠 **Long-term Memory**: Remembers player preferences and important facts using FAISS vector database
- 📚 **Knowledge Retrieval**: Searches for weapon recipes and lore using LanceDB with hybrid search
- 🛠️ **Inventory Management**: Adds crafted items to player inventories through game backend API
- 🔄 **WebSocket Communication**: Real-time interaction with players
- 📊 **Metrics Monitoring**: Prometheus metrics for token usage, conversation duration, and errors
- 🔧 **Configurable**: Use OpenAI GPT-4o or local Llama models

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/NPCango.git
   cd NPCango
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables (optional):
   ```bash
   export OPENAI_API_KEY=your_openai_api_key
   export NPC_NAME="Master Blacksmith"
   export NPC_DESCRIPTION="A legendary blacksmith known for crafting magical weapons."
   ```

## Quick Local Testing

You can quickly test the NPC functionality using the provided standalone test script:

1. Set your OpenAI API key:
   ```bash
   # On Linux/Mac:
   export OPENAI_API_KEY=your_openai_api_key
   
   # On Windows (Command Prompt):
   set OPENAI_API_KEY=your_openai_api_key
   
   # On Windows (PowerShell):
   $env:OPENAI_API_KEY="your_openai_api_key"
   ```

2. Make sure you have the OpenAI package installed:
   ```bash
   pip install openai>=1.0.0
   ```

3. Run the test script:
   ```bash
   cd NPCango
   python quick_npc_test.py
   ```

4. The script will send 4 test messages to the NPC and print the responses.

> **Note**: The test script is a standalone implementation that doesn't rely on the agno package or other parts of the codebase. It directly uses the OpenAI API to simulate the NPC functionality, making it easier to run without dependency issues.

### Expected Output

The test script will show:
- Debug logs of tool calls
- NPC responses to each message
- Stub inventory updates like: `[Stub] pretend to add 10x 钢锭 to player123`
- Lore search results with source references like:
  ```
  [1] 破甲长矛制作配方
  Relevance: 0.92
  制作破甲长矛需要以下材料：
  - 精炼钢锭 x5
  - 秘银合金 x2
  - 龙骨碎片 x1
  - 精工木柄 x1
  ```

This test verifies that:
1. The NPC can remember player names and recall them in later messages
2. The NPC can search for lore and include source references
3. The NPC can update player inventories (using the stub implementation)

## Running the System

### Start the Server

```bash
python -m NPCango.main
```

The server will start on http://localhost:8000 by default.

### Connect to the NPC

Connect to the WebSocket endpoint at `ws://localhost:8000/npc?player_id=<player_id>` to interact with the NPC.

Example using JavaScript:

```javascript
const socket = new WebSocket('ws://localhost:8000/npc?player_id=player123');

socket.onopen = () => {
  console.log('Connected to NPC');
  socket.send('Hello, blacksmith!');
};

socket.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('NPC response:', data.response);
};
```

### Demo Dialog

You can run a demo dialog using the included `demo_dialog()` function:

```bash
python -m NPCango.run
```

This will simulate a conversation with the blacksmith NPC.

## Configuration

The system can be configured using environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `NPC_NAME` | Name of the NPC | "Blacksmith" |
| `NPC_DESCRIPTION` | Description of the NPC | "A skilled blacksmith who can craft weapons and armor." |
| `MEMORY_PATH` | Path to store/load memory | "memory" |
| `LORE_DB_PATH` | Path to LanceDB database | "lancedb" |
| `API_BASE_URL` | Base URL for game API | "http://localhost:8000" |
| `MODEL` | Model to use | "gpt-4o" |
| `USE_LOCAL_LLAMA` | Whether to use local Llama | "false" |
| `LOCAL_LLAMA_URL` | URL for local Llama API | None |
| `MAX_TOKENS_PER_ROUND` | Maximum tokens per round | 6000 |
| `MAX_TOOL_CALLS` | Maximum tool calls per conversation | 4 |
| `MAX_LOOPS` | Maximum conversation loops | 10 |
| `DEBUG_MODE` | Whether to enable debug mode | "false" |

## Using Local Llama Instead of OpenAI

To use a local Llama model instead of OpenAI:

1. Set up a local Llama server (e.g., using [llama.cpp](https://github.com/ggerganov/llama.cpp) or [text-generation-webui](https://github.com/oobabooga/text-generation-webui))

2. Configure the environment variables:
   ```bash
   export USE_LOCAL_LLAMA=true
   export LOCAL_LLAMA_URL=http://localhost:8080/v1
   ```

3. Start the NPC system as usual:
   ```bash
   python -m NPCango.main
   ```

The system will now use your local Llama model instead of OpenAI.

## Adding New Tools

To add new tools to the NPC system:

1. Create a new function in an appropriate module (e.g., `tools.py`)

2. Register the tool in `agent_core.py`:

```python
@tool_registry.register
def my_new_tool(arg1: str, arg2: int) -> str:
    """
    Description of what the tool does.
    
    Args:
        arg1: Description of arg1
        arg2: Description of arg2
        
    Returns:
        Description of the return value
    """
    # Tool implementation
    return f"Result: {arg1}, {arg2}"
```

3. Update the system message in `build_npc()` to include instructions for using the new tool.

## Monitoring

The system exposes Prometheus metrics at the `/metrics` endpoint. Key metrics include:

- `npc_token_usage`: Token usage per conversation (prompt, completion, total)
- `npc_conversation_duration_seconds`: Duration of NPC conversations
- `npc_tool_calls_total`: Number of tool calls made by the NPC
- `tool_error_total`: Number of tool execution errors

You can use Prometheus and Grafana to monitor these metrics:

1. Add the NPC system to your Prometheus configuration:
   ```yaml
   scrape_configs:
     - job_name: 'npc'
       static_configs:
         - targets: ['localhost:8000']
   ```

2. Create dashboards in Grafana to visualize the metrics.

## Testing

Run the tests to verify the system is working correctly:

```bash
pytest
```

## License

MIT