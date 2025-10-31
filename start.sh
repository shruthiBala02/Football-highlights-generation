#!/bin/bash
# Install frontend dependencies and build React app
cd React
npm install
npm run build
cd ..

# Start FastAPI backend
cd football-backend
uvicorn main:app --host 0.0.0.0 --port 10000
