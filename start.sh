#!/bin/bash
set -e  # Exit immediately if any command fails
ROOT_DIR="$(dirname "$(realpath "$0")")"
echo "🚀 Starting build process from $ROOT_DIR"

# --------------------------
# 1️⃣  Build the frontend
# --------------------------
if [ -d "$ROOT_DIR/React" ]; then
  FRONTEND_DIR="$ROOT_DIR/React"
elif [ -d "$ROOT_DIR/react" ]; then
  FRONTEND_DIR="$ROOT_DIR/react"
else
  echo "❌ No React folder found. Expected 'React' or 'react' inside project root."
  exit 1
fi

echo "📦 Installing frontend dependencies in: $FRONTEND_DIR"
cd "$FRONTEND_DIR"

# Only install if package.json exists
if [ ! -f "package.json" ]; then
  echo "❌ package.json not found in React directory!"
  exit 1
fi

npm install --legacy-peer-deps
echo "🏗️  Building React app..."
npm run build
echo "✅ React build completed successfully."

# --------------------------
# 2️⃣  Start the backend
# --------------------------
cd "$ROOT_DIR/football-backend"

# Check backend main file
if [ ! -f "main.py" ]; then
  echo "❌ Could not find main.py in football-backend!"
  exit 1
fi

echo "🎯 Starting FastAPI backend with uvicorn..."
exec uvicorn main:app --host 0.0.0.0 --port 10000
