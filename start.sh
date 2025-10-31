#!/bin/bash
set -e

echo "🚀 Starting build process..."

# Build frontend (works for 'React' or 'react')
if [ -d "React" ]; then
  cd React
elif [ -d "react" ]; then
  cd react
else
  echo "❌ React folder not found!"
  exit 1
fi

echo "📦 Installing frontend dependencies..."
npm install

echo "🏗️  Building React app..."
npm run build
cd ..

# Start backend
cd football-backend
echo "🎯 Starting FastAPI backend..."
uvicorn main:app --host 0.0.0.0 --port 10000
