Param(
    [string]$ImageName = "maxamed1205/ultramotion-igt-inference:latest",
    [int]$HostPort = 18945
)

Write-Host "Running Docker image: $ImageName"

docker run --gpus all -it --rm -p ${HostPort}:18945 --name umi_service $ImageName
