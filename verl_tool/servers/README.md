## Tool Server

The tool server provides sandboxed execution environments for Metis's agent during training and inference.

### Quick Start

```bash
# Start the tool server
host=localhost
port=30569
tool_type=metis
workers_per_tool=32
python -m verl_tool.servers.serve --host $host --port $port --tool_type $tool_type --workers_per_tool $workers_per_tool
```

Or use the convenience script:

```bash
bash examples/train/start_tool_server.sh [PORT] [WORKERS]
```

### Available Tools

| Tool | Type | Description |
|------|------|-------------|
| `metis` | Full tool | Python execution + text search + image search |
| `metis_code` | Code-only | Python execution only (no search) |

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SEARCH_PROVIDER` | Search backend (`serpapi`, `brightdata`, `serper.dev`) | `brightdata` |
| `SERPER_API_KEY` | API key for Serper.dev search | `""` |
| `BRIGHTDATA_API_TOKEN` | API token for BrightData search | `""` |
| `METIS_SESSION_DIR` | Directory for tool execution sessions | `/tmp/metis_sessions` |

### API Endpoints

- `POST /get_observation` — Execute a tool action and return the observation
- `GET /health` — Health check
- `GET /metrics` — Server metrics

### Testing

```bash
# Test the metis tool server
python -m verl_tool.servers.tests.test_metis_tool metis --url=http://localhost:30569/get_observation
```
