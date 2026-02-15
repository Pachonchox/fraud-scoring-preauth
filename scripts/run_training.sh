#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [ ! -d ".venv" ]; then
  echo "[ERROR] No se encontró .venv. Crea el entorno primero:"
  echo "  ./scripts/create_venv.sh"
  exit 1
fi

if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" || "$OSTYPE" == "cygwin" ]]; then
  source .venv/Scripts/activate
else
  source .venv/bin/activate
fi

echo "[INFO] Entrenando modelos (RandomForest + HistGBM + MLP) para score pre-autorizacion..."
python -m src.models.train_pipeline
