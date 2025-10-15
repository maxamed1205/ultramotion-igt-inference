Param(
    [string]$ImageName = "maxamed1205/ultramotion-igt-inference:latest"
)

Write-Host "Building Docker image: $ImageName"

docker build -t $ImageName ..

if ($LASTEXITCODE -ne 0) {
    Write-Error "Docker build failed"
    exit $LASTEXITCODE
}

Write-Host "Build completed: $ImageName"
