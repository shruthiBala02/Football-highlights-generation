#!/bin/bash
set -e  # Exit immediately on error
ROOT_DIR="$(dirname "$(realpath "$0")")"
echo "🚀 Starting build process from $ROOT_DIR"

# --------------------------
# 1️⃣ Build React frontend
# --------------------------
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

  echo "🏗️  Building React app..."
  npm run build
  echo "✅ React build completed successfully."
  cd "$ROOT_DIR"
else
  echo "❌ React app directory not found at React/football-highlights!"
  exit 1
fi

# --------------------------
# 2️⃣ Start FastAPI backend
# --------------------------
BACKEND_DIR="$ROOT_DIR/football-backend"

if [ -d "$BACKEND_DIR" ]; then
  cd "$BACKEND_DIR"

  if [ ! -f "main.py" ]; then
    echo "❌ main.py not found in $BACKEND_DIR"
    exit 1
  fi

  echo "🎯 Starting FastAPI backend..."
  exec uvicorn main:app --host 0.0.0.0 --port 10000
else
  echo "❌ Backend directory football-backend not found!"
  exit 1
fi
