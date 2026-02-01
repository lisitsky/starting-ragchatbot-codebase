# Code Quality Tooling — Changes Log

## What was added

### black formatter
- Added `black>=24.0.0` to the `[dependency-groups] dev` list in `pyproject.toml`
- Added `[tool.black]` configuration block:
  - `line-length = 100` — wider than the default 88 to reduce noisy wrapping on typical backend lines
  - `target-version = ["py313"]` — matches the project's `requires-python = ">=3.13"`

### check_quality.sh
- New script at the project root for running quality checks from the command line
- **Check mode** (default): `./check_quality.sh` — runs `black --check`, exits non-zero if any file needs formatting
- **Fix mode**: `./check_quality.sh --fix` — runs `black` in place, auto-formats all files

## Files reformatted by black (initial pass)

All 13 Python files in the project were formatted in place:

| File | Notes |
|------|-------|
| `main.py` | Already clean — left unchanged |
| `backend/config.py` | Trailing whitespace and blank lines normalized |
| `backend/models.py` | Trailing whitespace removed |
| `backend/session_manager.py` | Trailing whitespace on class/method bodies |
| `backend/ai_generator.py` | Long string literals and dict literals re-wrapped |
| `backend/app.py` | Duplicate imports consolidated by formatter; trailing whitespace removed |
| `backend/rag_system.py` | Trailing whitespace on docstrings and method bodies |
| `backend/search_tools.py` | Trailing whitespace removed throughout |
| `backend/vector_store.py` | Trailing whitespace and blank-line normalization |
| `backend/document_processor.py` | Extra blank lines between methods collapsed |
| `backend/tests/__init__.py` | Already clean — left unchanged |
| `backend/tests/conftest.py` | Trailing whitespace removed |
| `backend/tests/test_01_environment.py` | Long assert continuation lines re-wrapped |
| `backend/tests/test_04_ai_generator.py` | Trailing whitespace removed |
| `backend/tests/test_07_error_injection.py` | Trailing whitespace removed |

## How to use going forward

```bash
# Verify all files are formatted (CI / pre-commit friendly)
./check_quality.sh

# Auto-fix any formatting drift
./check_quality.sh --fix
```
