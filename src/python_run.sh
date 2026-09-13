#!/bin/bash

set -e

if [ $# -lt 1 ]; then
    echo "Usage: ./python_run <script.py> [args...]"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_DIR="$(dirname "$SRC_DIR")"

source "$PROJECT_DIR/venv/bin/activate"

export PYTHONPATH="$SRC_DIR${PYTHONPATH:+:$PYTHONPATH}"

SCRIPT="$1"
shift

python "$SCRIPT_DIR/$SCRIPT" "$@"
