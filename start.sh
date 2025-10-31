#!/bin/bash
set -e  # Exit on any error

# === Resolve root directory ===
ROOT_DIR="$(dirname "$(realpath "$0")")"
echo "🚀 Starting build process from $ROOT_DIR"

# ============================================================
# 1️⃣  BUILD REACT FRONTEND (inside React/football-highlights)
# ============================================================
FRONTEND_DIR="$ROOT_DIR/React/football-highlights"

if [ -d "$FRONTEND_DIR" ]; then
  echo "📦 Found React app at: $FRONTEND_DIR"
  cd "$FRONTEND_DIR"

  if [ ! -f "package.json" ]; then
    echo "❌ package.json not found in $FRONTEND_DIR"
    exit 1
  fi

  echo "📦 Installing frontend dependencies..."
  npm install --legacy-peer-deps

  echo "🔧 Fixing react-scripts permission issues..."
  chmod +x ./node_modules/.bin/react-scripts || true

  echo "🏗️  Building React app safely..."
  node node_modules/react-scripts/bin/react-scripts.js build

  echo "✅ React build completed successfully."
  cd "$ROOT_DIR"
else
  echo "❌ React app directory not found at React/football-highlights!"
  exit 1
fi

# ============================================================
# 2️⃣  START FASTAPI BACKEND
# ============================================================
BACKEND_DIR="$ROOT_DIR/football-backend"

if [ -d "$BACKEND_DIR" ]; then
  cd "$BACKEND_DIR"

  if [ ! -f "main.py" ]; then
    echo "❌ main.py not found in $BACKEND_DIR"
    exit 1
  fi

  echo "🎯 Starting FastAPI backend..."
  # Ensure proper permissions for uvicorn
  chmod +x "$(which uvicorn)" || true
  PORT=${PORT:-10000}
exec uvicorn main:app --host 0.0.0.0 --port $PORT
else
  echo "❌ Backend directory football-backend not found!"
  exit 1
fi
