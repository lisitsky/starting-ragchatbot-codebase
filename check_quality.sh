#!/bin/bash
# Code quality check script for the RAG chatbot project.
# Runs black in check mode (no changes made) and reports formatting status.
#
# Usage:
#   ./check_quality.sh          # check only
#   ./check_quality.sh --fix    # auto-format files in place

set -e

TARGETS="main.py backend/"
FIX_MODE=false

if [ "$1" = "--fix" ]; then
    FIX_MODE=true
fi

echo "=== Code Quality Check ==="
echo ""

# --- black ---
if [ "$FIX_MODE" = true ]; then
    echo "[black] Formatting..."
    uv run black $TARGETS
else
    echo "[black] Checking formatting..."
    if uv run black --check $TARGETS; then
        echo "[black] All files are formatted correctly."
    else
        echo ""
        echo "[black] FAILED — run with --fix to auto-format:"
        echo "  ./check_quality.sh --fix"
        exit 1
    fi
fi

echo ""
echo "=== All checks passed ==="
