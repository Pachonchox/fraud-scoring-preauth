#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN=${PYTHON_BIN:-python3}

echo "[INFO] Creando entorno virtual en .venv usando: $PYTHON_BIN"
$PYTHON_BIN -m venv .venv

echo "[INFO] Activando entorno virtual"
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" || "$OSTYPE" == "cygwin" ]]; then
  # Windows (Git Bash / Cygwin)
  source .venv/Scripts/activate
else
  source .venv/bin/activate
fi

echo "[INFO] Instalando dependencias desde requirements.txt"
pip install --upgrade pip
pip install -r requirements.txt

echo "[OK] Entorno virtual listo. Para activarlo manualmente:"
echo "  source .venv/bin/activate   # Linux / macOS"
echo "  .venv\\Scripts\\activate    # Windows (PowerShell/cmd)"

