#!/bin/bash
# EdgeOps AI Platform - Automated Multi-Service Launcher (Linux / macOS)
set -e

echo "=========================================================="
echo "   ⚡ EdgeOps AI Platform: Launching Multi-Service Stack   "
echo "=========================================================="

# 1. Start Backend API in background
echo "[1/2] Starting EdgeOps FastAPI Backend on http://127.0.0.1:8000..."
python -m uvicorn api:app --host 127.0.0.1 --port 8000 --reload &
BACKEND_PID=$!

sleep 3

# 2. Start Frontend Vite Server in background
echo "[2/2] Starting EdgeOps React Dashboard on http://localhost:5173..."
cd edgeops-dashboard
npm run dev &
FRONTEND_PID=$!
cd ..

trap "kill $BACKEND_PID $FRONTEND_PID" EXIT

echo "=========================================================="
echo "🚀 EdgeOps AI Platform is running!"
echo " - FastAPI Swagger: http://127.0.0.1:8000/docs"
echo " - React Dashboard: http://localhost:5173"
echo "=========================================================="

wait
