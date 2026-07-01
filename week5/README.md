# Week 4 — Agents & Tool Calling

## Architecture
User question -> agent loop -> LLM picks tool(s) -> execute tool -> feed result back -> LLM answers
                                     |
                                     +-- get_current_time
                                     +-- calculate
                                     +-- get_weather (open-meteo API)
                                     +-- search_knowledge_base (RAG over docs.txt)

## Files
- `tools.py`        - tool functions + schemas (TOOLS) + name->fn map (AVAILABLE)
- `agent.py`        - the agent loop with dedup guard + system prompt
- `main.py`         - FastAPI app with /agent endpoint + session memory
- `tool_test.py`    - standalone tool-calling experiments
- `weather_test.py` - external API tool demo
- `agent_full.py`   - full agent demo (run directly)
- `chat-ui/`        - React UI with RAG/Agent mode toggle + tool-steps display

## Running locally
Terminal 1: ollama serve
Terminal 2: source venv/bin/activate && uvicorn main:app --reload
Terminal 3: cd chat-ui && npm start

## Endpoints
GET  /health      - health check + chunk count
POST /search      - semantic search only
POST /rag         - full RAG (retrieve + generate)
POST /rag/stream  - streaming RAG
POST /chat        - direct LLM (no RAG)
POST /chat/stream - streaming direct LLM
POST /agent       - agentic tool-calling with conversation memory

## Tools
| Tool                  | Purpose                          | Args                  |
|-----------------------|----------------------------------|-----------------------|
| get_current_time      | Current date/time                | none                  |
| calculate             | Basic math evaluation            | expression            |
| get_weather           | Live weather via open-meteo API  | latitude, longitude   |
| search_knowledge_base | RAG search over docs.txt         | query                 |

## Key learnings
- Tool calling: LLM returns a structured tool_calls request, not an answer
- The agent loop runs until the model produces an answer with no tool calls
- max_steps cap prevents infinite loops
- Dedup guard prevents the model re-calling the same tool with same args
- System prompt nudges model to use exact tool outputs (fixes hallucinated values)
- Conversation memory via session_id enables multi-turn context
- Small models (3b/8b) on CPU are slow for agents — GPU (A10) or hosted inference fixes this

## Notes
- Uses llama3.1:8b for better tool selection (llama3.2:3b is faster but less reliable)
- Agent mode is slower than RAG: multiple sequential LLM calls per question
- Will deploy with GPU acceleration (A10 on OKE) or OCI Generative AI in Week 5
