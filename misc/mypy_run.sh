#!/bin/bash

set -e

if [ $# -lt 1 ]; then
    echo "Usage: ./mypy_run <script.py> [args...]"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
SRC_DIR="${PROJECT_DIR}/src"

source "$PROJECT_DIR/venv/bin/activate"

cd "$PROJECT_DIR"

mypy "$SCRIPT_DIR/$1"