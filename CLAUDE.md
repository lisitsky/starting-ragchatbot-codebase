# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A full-stack RAG (Retrieval-Augmented Generation) chatbot for querying course materials. Uses FastAPI backend with ChromaDB vector storage, Anthropic's Claude API with tool-calling, and a vanilla JavaScript frontend.

## Development Commands

### Setup
```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Create .env file with API key
echo "ANTHROPIC_API_KEY=sk-ant-api03-..." > .env
```

### Running the Application
```bash
# Quick start (recommended)
./run.sh

# Manual start
cd backend && uv run uvicorn app:app --reload --port 8000

# Access points:
# - Web UI: http://localhost:8000
# - API Docs: http://localhost:8000/docs
```

### Development
```bash
# Run with specific port
cd backend && uv run uvicorn app:app --reload --port 8001

# Clear and rebuild vector database
# Delete ./backend/chroma_db directory, then restart server
```

## Dependency Management

**IMPORTANT: This project uses `uv` for all dependency management. DO NOT use `pip` directly.**

```bash
# Add a new dependency
uv add <package-name>

# Add a development dependency
uv add --dev <package-name>

# Remove a dependency
uv remove <package-name>

# Update dependencies
uv sync

# Lock dependencies (updates uv.lock)
uv lock
```

Dependencies are managed via:
- `pyproject.toml` - declares dependencies and project metadata
- `uv.lock` - locks exact versions for reproducibility

**Never use**: `pip install`, `pip freeze`, `requirements.txt` - these bypass uv's dependency resolution.

## Architecture Overview

### RAG Pipeline Flow
```
User Query → FastAPI → RAGSystem → AIGenerator → Claude API (call #1 with tools)
    → Claude decides to search → ToolManager → CourseSearchTool
    → VectorStore → ChromaDB (semantic search)
    → Search Results → Claude API (call #2 with context)
    → Final Answer + Sources → SessionManager (history) → Frontend
```

### Key Architectural Patterns

**1. Two-Phase Claude API Pattern**
- **First call**: Claude receives user query + search tool definitions, decides to invoke search
- **Second call**: Claude receives search results, synthesizes final answer
- No tools provided in second call (prevents infinite loops)
- Implementation: `backend/ai_generator.py:43-135`

**2. Dual ChromaDB Collections**
- `course_catalog`: Course metadata (title, instructor, lessons) - used for course name resolution
- `course_content`: Actual content chunks with embeddings - used for semantic search
- Course names resolved via semantic search before content search (`vector_store.py:102-116`)

**3. Tool-Use Architecture**
- Tools implement abstract `Tool` class (`search_tools.py:6-17`)
- `ToolManager` handles registration and execution (`search_tools.py:116-154`)
- `CourseSearchTool` tracks sources in `self.last_sources` for UI display
- Sources extracted after query completes, then reset (`rag_system.py:129-133`)

**4. Session-Based Conversation History**
- Each user gets unique session ID (auto-generated or provided)
- History limited to last 2 exchanges (4 messages) via `MAX_HISTORY=2`
- History injected into Claude's system prompt on subsequent queries
- In-memory storage (lost on restart) - `session_manager.py`

**5. Document Processing Pipeline**
- Expects structured format: `Course Title:`, `Course Link:`, `Course Instructor:`, then `Lesson N:` markers
- Sentence-based chunking (800 chars) with 100-char overlap to preserve context
- Context injection: Prepends "Course X Lesson Y content:" to chunks for better retrieval
- Implementation: `document_processor.py:97-259`

### Critical Configuration (`backend/config.py`)

```python
ANTHROPIC_MODEL = "claude-sonnet-4-20250514"  # Model used
CHUNK_SIZE = 800           # Text chunk size (characters)
CHUNK_OVERLAP = 100        # Overlap between chunks
MAX_RESULTS = 5            # Search results per query
MAX_HISTORY = 2            # Conversation exchanges stored
CHROMA_PATH = "./chroma_db"  # Vector DB location
```

### Data Models (`backend/models.py`)

- **Course**: Container for course metadata + lessons (title is unique ID)
- **Lesson**: lesson_number, title, optional lesson_link
- **CourseChunk**: Content chunk with course_title, lesson_number, chunk_index

### Component Responsibilities

| Component | Responsibility | Critical Files |
|-----------|---------------|----------------|
| `RAGSystem` | Orchestrates entire query pipeline | `rag_system.py` |
| `DocumentProcessor` | Parses course files, chunks text | `document_processor.py:97-259` |
| `VectorStore` | ChromaDB wrapper, semantic search | `vector_store.py:61-100` |
| `AIGenerator` | Claude API interaction, tool execution | `ai_generator.py:43-135` |
| `ToolManager` | Tool registration and routing | `search_tools.py:116-154` |
| `SessionManager` | Conversation history (in-memory) | `session_manager.py` |

### Important Implementation Details

**Startup Process (`backend/app.py:88-98`)**
- `@app.on_event("startup")` loads all documents from `../docs`
- Skips already-indexed courses (checks `vector_store.get_existing_course_titles()`)
- Processing happens synchronously - startup completes when all docs indexed

**Search Tool Context Enhancement (`search_tools.py:88-114`)**
- Results formatted with `[Course Title - Lesson N]` headers
- Sources tracked in `self.last_sources` as list of strings
- Retrieved via `tool_manager.get_last_sources()` after query completes
- Must be reset with `tool_manager.reset_sources()` to prevent carryover

**Vector Search Filter Building (`vector_store.py:118-133`)**
- Supports AND combinations: `{"$and": [{"course_title": X}, {"lesson_number": Y}]}`
- Course name resolution: User can provide partial name (e.g., "MCP"), semantic search finds best match
- No filter = search all content

**Frontend Session Handling (`frontend/script.js:45-96`)**
- `currentSessionId` stored globally, sent with each query
- First query: `session_id: null`, backend creates session and returns ID
- Subsequent queries: use stored session ID for conversation context
- Loading animation: creates temp element, removed when response arrives

### Course Document Format

```
Course Title: [title]
Course Link: [url]
Course Instructor: [name]

Lesson 0: [lesson title]
Lesson Link: [url]
[lesson content...]

Lesson 1: [next lesson title]
...
```

Files in `docs/` folder are auto-loaded on startup. Supports `.txt`, `.pdf`, `.docx` extensions.

### Modifying Core Behavior

**Change chunk size**: Edit `CHUNK_SIZE` in `config.py` → delete `chroma_db/` → restart
**Change Claude model**: Edit `ANTHROPIC_MODEL` in `config.py` → restart
**Add new tools**: Create class extending `Tool` → register with `tool_manager.register_tool()`
**Modify system prompt**: Edit `AIGenerator.SYSTEM_PROMPT` in `ai_generator.py:8-30`
**Increase conversation memory**: Edit `MAX_HISTORY` in `config.py` (default 2 exchanges = 4 messages)

### Common Gotchas

1. **Dependency management**: ALWAYS use `uv add/remove/sync` - NEVER use `pip install` directly
2. **Sources not displaying**: Check `search_tools.py:112` - sources must be stored in `self.last_sources`
3. **Chunk context inconsistency**: Line 186 adds "Lesson N content:" but line 234 adds "Course X Lesson Y content:" (intentional variation for first vs last lesson)
4. **Session persistence**: Sessions are in-memory only - lost on server restart
5. **Course name matching**: Uses semantic search on course catalog, not exact string matching
6. **Startup time**: Proportional to number of documents in `docs/` - embedding generation is CPU-bound
