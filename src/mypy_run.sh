#!/bin/bash

set -e

if [ $# -lt 1 ]; then
    echo "Usage: ./mypy_run <script.py> [args...]"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_DIR="$(dirname "$SRC_DIR")"

source "$PROJECT_DIR/venv/bin/activate"

cd "$PROJECT_DIR"

mypy "$SCRIPT_DIR/$1"