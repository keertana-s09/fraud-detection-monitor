#!/bin/bash
# Start the fraud detection dashboard
# Usage: ./start.sh
#
# Opens a landing page where you can:
#   - Upload your own transaction CSV
#   - Or click "Use Demo Data" for simulated data with concept drift
#
# The model trains after you provide data (~60-90s), then the dashboard loads.

set -e

cd "$(dirname "$0")"

echo "=== Fraud Detection Monitor ==="
echo ""

# Build frontend if needed
if [ ! -d "frontend/dist" ] || [ "frontend/src/App.jsx" -nt "frontend/dist/index.html" ]; then
    echo "[1/2] Building React frontend..."
    cd frontend
    npm install --silent 2>/dev/null
    npm run build 2>/dev/null
    cd ..
    echo "      Frontend built."
else
    echo "[1/2] Frontend already built."
fi

echo "[2/2] Starting server..."
echo ""
echo "      Dashboard: http://localhost:8000"
echo ""

python3 -m uvicorn api:app --host 0.0.0.0 --port 8000
