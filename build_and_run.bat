@echo off
echo Building Docker images...
docker-compose build

echo Starting containers...
docker-compose up -d

echo.
echo Services running!
echo Streamlit: http://localhost:8501
echo FastAPI: http://localhost:8000/docs
pause