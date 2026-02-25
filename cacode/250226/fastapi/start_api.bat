@echo off
echo Starting Clinical LLM Enrichment API...
echo.
echo API will be available at: http://localhost:8000
echo Interactive docs at: http://localhost:8000/docs
echo.
uvicorn main:app --reload --host 0.0.0.0 --port 8000
