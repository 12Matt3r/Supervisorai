#!/bin/sh
# STDIO mode startup script - suitable for local tool integration
set -e

# Change to script directory
cd "$(dirname "$0")"

# Create independent virtual environment (if it doesn't exist)
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..." >&2
    uv venv
    echo "Installing dependencies..." >&2
    echo "Note: Dependency installation may take several minutes. Please wait..." >&2
    uv sync
fi

# Check necessary environment variables
if [ -z "$SUPERVISOR_DATA_DIR" ]; then
    echo "Info: SUPERVISOR_DATA_DIR not set, using default ./supervisor_data" >&2
fi

if [ -z "$LOG_LEVEL" ]; then
    echo "Info: LOG_LEVEL not set, using default INFO" >&2
fi

# Start STDIO mode MCP server
uv run src/server/main.py