#!/bin/bash
# Detect folder name for React build
if [ -d "React" ]; then
  cd React
elif [ -d "react" ]; then
  cd react
else
  echo "❌ React folder not found!"
  exit 1
fi

# Build frontend
npm install
npm run build
cd ..

# Start backend
cd football-backend
uvicorn main:app --host 0.0.0.0 --port 10000
