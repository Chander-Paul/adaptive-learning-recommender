#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV_PY="$ROOT_DIR/.venv/bin/python"
if [[ ! -x "$VENV_PY" ]]; then
    echo "Missing virtual environment python at $VENV_PY"
    echo "Create it first, then run this script again."
    exit 1
fi

echo "Removing existing SQLite DB files..."
rm -f "$ROOT_DIR/data/learner_recommender.db" "$ROOT_DIR/app/data/learner_recommender.db"

echo "Applying migrations from a clean state..."
export FLASK_APP="app/flask_db.py"
"$VENV_PY" -m flask db upgrade

echo "Done. Fresh database created via migrations."
