# ============================
#   MULTI-STAGE BUILD
# ============================

# Stage 1: Build React frontend
FROM node:20 AS frontend
WORKDIR /app/React/football-highlights
COPY React/football-highlights/package*.json ./
RUN npm install --legacy-peer-deps
COPY React/football-highlights/ .
RUN npm run build

# Stage 2: FastAPI backend with built frontend
FROM python:3.10-slim
WORKDIR /app

# Copy backend
COPY football-backend ./football-backend

# Copy root start.sh
COPY start.sh ./start.sh

# Copy built frontend into backend
RUN mkdir -p ./football-backend/frontend_build
COPY --from=frontend /app/React/football-highlights/build ./football-backend/frontend_build

# Install backend dependencies
COPY football-backend/requirements.txt ./football-backend/
RUN pip install --no-cache-dir -r football-backend/requirements.txt

# Ensure start.sh is executable
RUN chmod +x ./start.sh

# Expose backend port
EXPOSE 8000

# Run from root
CMD ["bash", "./start.sh"]
