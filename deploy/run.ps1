# STRIAX - run the pipeline once on ECS Fargate, wait for it, and show the logs.
#
# Usage (from the repo root):
#   powershell -ExecutionPolicy Bypass -File .\deploy\run.ps1               # daily: fetch + predict
#   powershell -ExecutionPolicy Bypass -File .\deploy\run.ps1 -Command all  # fetch + full retrain + predict
#
# The task runs in the account's default VPC with a public IP (no NAT gateway needed),
# then shuts down. You only pay for the minutes it runs.

param(
    [ValidateSet("daily", "all", "fetch", "train", "predict")]
    [string]$Command = "daily",
    [string]$Region = "us-east-1",
    [string]$Name = "striax",
    [switch]$NoWait
)

function Fail($msg) { Write-Host "ERROR: $msg" -ForegroundColor Red; exit 1 }
function Step($msg) { Write-Host "`n== $msg" -ForegroundColor Cyan }
function Ok($msg)   { Write-Host "   $msg" -ForegroundColor Green }

# ---------------------------------------------------------------------------
Step "Finding the default network"
$vpc = aws ec2 describe-vpcs --filters "Name=isDefault,Values=true" --region $Region --query "Vpcs[0].VpcId" --output text
if ($LASTEXITCODE -ne 0 -or -not $vpc -or $vpc.Trim() -eq "None") { Fail "No default VPC in $Region." }
$vpc = $vpc.Trim()

$subnets = aws ec2 describe-subnets --filters "Name=vpc-id,Values=$vpc" "Name=default-for-az,Values=true" --region $Region --query "Subnets[].SubnetId" --output text
$subnetList = (($subnets -split "\s+") | Where-Object { $_ }) -join ","
if (-not $subnetList) { Fail "No default subnets found in $vpc." }

$sg = aws ec2 describe-security-groups --filters "Name=vpc-id,Values=$vpc" "Name=group-name,Values=default" --region $Region --query "SecurityGroups[0].GroupId" --output text
$sg = $sg.Trim()
Ok "vpc $vpc, security group $sg"

# ---------------------------------------------------------------------------
Step "Starting Fargate task (command: $Command)"
$net = "awsvpcConfiguration={subnets=[$subnetList],securityGroups=[$sg],assignPublicIp=ENABLED}"

$ovFile = Join-Path $env:TEMP "striax-overrides.json"
$ovJson = '{"containerOverrides":[{"name":"' + $Name + '","command":["' + $Command + '"]}]}'
[System.IO.File]::WriteAllText($ovFile, $ovJson, (New-Object System.Text.UTF8Encoding($false)))

$resp = aws ecs run-task --cluster $Name --launch-type FARGATE --task-definition $Name --network-configuration $net --overrides "file://$ovFile" --region $Region --output json | Out-String | ConvertFrom-Json
if (-not $resp -or $resp.failures.Count -gt 0 -or $resp.tasks.Count -eq 0) {
    if ($resp) { $resp.failures | Format-List | Out-String | Write-Host }
    Fail "Task did not start."
}
$taskArn = $resp.tasks[0].taskArn
$taskId = $taskArn.Split("/")[-1]
Ok "task $taskId started"

if ($NoWait) {
    Write-Host "   Not waiting. Logs later: aws logs tail /ecs/$Name --since 1h --region $Region"
    exit 0
}

# ---------------------------------------------------------------------------
Step "Waiting for it to finish (usually 2-5 minutes)"
aws ecs wait tasks-stopped --cluster $Name --tasks $taskArn --region $Region
if ($LASTEXITCODE -ne 0) { Write-Host "   Still running after 10 minutes; check the logs below or the ECS console." -ForegroundColor Yellow }

$task = (aws ecs describe-tasks --cluster $Name --tasks $taskArn --region $Region --output json | Out-String | ConvertFrom-Json).tasks[0]
$exitCode = $task.containers[0].exitCode

Step "Logs"
aws logs tail "/ecs/$Name" --log-stream-names "$Name/$Name/$taskId" --since 1h --region $Region

Step "Result"
if ($exitCode -eq 0) {
    Ok "Pipeline finished successfully (exit code 0)."
    aws s3 ls "s3://$Name-$((aws sts get-caller-identity --query Account --output text).Trim())/out/" --region $Region
} else {
    Write-Host "   Exit code: $exitCode" -ForegroundColor Red
    Write-Host "   Stop reason: $($task.stoppedReason)" -ForegroundColor Red
    if ($task.containers[0].reason) { Write-Host "   Container reason: $($task.containers[0].reason)" -ForegroundColor Red }
    exit 1
}
