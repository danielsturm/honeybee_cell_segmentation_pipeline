#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="$(command -v python3 || command -v python)"

if [[ "$(uname)" == "Linux" ]] && command -v apt-get &>/dev/null; then
  PYTHON_VERSION=$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
  if ! dpkg -s "python${PYTHON_VERSION}-venv" &>/dev/null; then
    echo "python${PYTHON_VERSION}-venv is not installed."
    echo "Please run: sudo apt-get install python${PYTHON_VERSION}-venv"
    exit 1
  fi
fi

if [ ! -d .venv ]; then
  "$PYTHON_BIN" -m venv .venv
fi

source .venv/bin/activate

python -m pip install --upgrade pip

python -m pip install poetry

export POETRY_VIRTUALENVS_CREATE=false
export POETRY_NO_INTERACTION=1

poetry install

if [[ "$(uname)" == "Linux" ]]; then
  QXCB_SO=$(find .venv -name libqxcb.so | head -n 1)
  if [ -n "$QXCB_SO" ]; then
    echo "Checking Qt platform plugin dependencies..."
    MISSING_LIBS=$(ldd "$QXCB_SO" | grep "not found" || true)
    if [ -n "$MISSING_LIBS" ]; then
      echo "Some Qt dependencies are missing:"
      echo "$MISSING_LIBS"
      echo "You may need to install these packages"
      echo "Refer to the missing .so names above for exact packages."
      exit 1
    else
      echo "All Qt platform plugin dependencies are satisfied."
    fi
  fi
fi

echo "Done. Please run: source .venv/bin/activate"
echo "Then you can run the annotator with: poetry run annotator <path/to/images>"