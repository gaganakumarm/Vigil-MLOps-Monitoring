Write-Host "🚀 Starting MLOps Monitoring Project..." -ForegroundColor Green

# Build and start containers
docker compose up --build -d

Write-Host "`n Containers are up and running!"
Write-Host "-------------------------------------"
Write-Host "API running at: http://localhost:8000"
Write-Host "Dashboard running at: http://localhost:8501"
Write-Host "-------------------------------------"
Write-Host "Open your browser manually to view the dashboard."
Write-Host "To stop the containers, run: docker compose down"
