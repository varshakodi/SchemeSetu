#!/usr/bin/env bash
# One command to demo SchemeSetu. Checks every prerequisite first, so problems
# surface here rather than in front of an interviewer.
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
[ -x "$PY" ] || { echo "No virtualenv. Run: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"; exit 1; }
"$PY" scripts/doctor.py "$@"
echo "Opening http://localhost:8501  (press Ctrl+C to stop)"
exec "$PY" -m streamlit run app.py
