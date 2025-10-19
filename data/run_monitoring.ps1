# Navigate to project folder
Set-Location "C:\Users\gagan\OneDrive\Desktop\automated-mlops-monitoring"

# Prefer 'docker compose'; fall back to 'docker-compose' if needed
$dockerCompose = "docker compose"
if (-not (Get-Command "docker" -ErrorAction SilentlyContinue)) {
  Write-Error "Docker is not installed or not on PATH."
  exit 1
}
# Check if 'docker compose' is available (suppress stderr cleanly)
$null = (& docker compose version) 2> $null
$hasComposeV2 = ($LASTEXITCODE -eq 0)
if (-not $hasComposeV2) {
  if (Get-Command "docker-compose" -ErrorAction SilentlyContinue) {
    $dockerCompose = "docker-compose"
  } else {
    Write-Error "Neither 'docker compose' nor 'docker-compose' is available."
    exit 1
  }
}

# Start services
Invoke-Expression "$dockerCompose up -d"

# Wait for 'api' container to be up (retry loop)
$retries = 12
$delaySec = 5
for ($i = 1; $i -le $retries; $i++) {
  $running = (& docker ps --filter "name=_api_1" --filter "status=running" --format "{{.Names}}")
  if ($running) { break }
  Start-Sleep -Seconds $delaySec
}
if (-not $running) {
  Write-Warning "API container not confirmed running; attempting exec anyway."
}

# Run the monitoring job inside the API container (no-TTY for CI safety)
Invoke-Expression "$dockerCompose exec -T api python monitoring_job.py"