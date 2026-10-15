# STRIAX - one-time AWS setup for running the pipeline on ECS Fargate.
#
# Creates (or reuses, if they already exist):
#   - CloudWatch log group  /ecs/striax          (container output, kept 30 days)
#   - IAM role  striax-execution-role             (lets ECS pull the image from ECR and write logs)
#   - IAM role  striax-task-role                  (lets the code read/write ONLY the striax S3 bucket)
#   - ECS cluster  striax
#   - ECS task definition  striax                 (image, CPU/memory, S3 settings)
#
# Safe to re-run. Prerequisites: AWS CLI configured, image pushed to ECR, S3 bucket created.
#
# Usage (from the repo root):
#   powershell -ExecutionPolicy Bypass -File .\deploy\setup.ps1

param(
    [string]$Region = "us-east-1",
    [string]$Name = "striax",
    [ValidateSet("", "X86_64", "ARM64")]
    [string]$Arch = ""
)

$here = Split-Path -Parent $MyInvocation.MyCommand.Path

function Fail($msg) { Write-Host "ERROR: $msg" -ForegroundColor Red; exit 1 }
function Step($msg) { Write-Host "`n== $msg" -ForegroundColor Cyan }
function Ok($msg)   { Write-Host "   $msg" -ForegroundColor Green }

function Write-NoBom($path, $text) {
    [System.IO.File]::WriteAllText($path, $text, (New-Object System.Text.UTF8Encoding($false)))
}

# Fill __PLACEHOLDERS__ in a template from this folder and save it to a temp file.
function Render($template, $vars) {
    $text = Get-Content -Raw -Path (Join-Path $here $template)
    foreach ($k in $vars.Keys) { $text = $text.Replace("__${k}__", $vars[$k]) }
    $out = Join-Path $env:TEMP ("striax-" + $template)
    Write-NoBom $out $text
    return $out
}

function Ensure-Role($role, $trustFile) {
    aws iam get-role --role-name $role > $null 2>&1
    if ($LASTEXITCODE -eq 0) { Ok "role $role already exists"; return }
    aws iam create-role --role-name $role --assume-role-policy-document "file://$trustFile" > $null
    if ($LASTEXITCODE -ne 0) { Fail "could not create role $role" }
    Ok "created role $role"
    $script:createdRole = $true
}

# ---------------------------------------------------------------------------
Step "Checking prerequisites"

$Account = aws sts get-caller-identity --query Account --output text
if ($LASTEXITCODE -ne 0 -or -not $Account) { Fail "AWS CLI is not configured. Run 'aws configure' first." }
$Account = $Account.Trim()
$Bucket = "$Name-$Account"
Ok "account $Account, region $Region"

aws s3api head-bucket --bucket $Bucket > $null 2>&1
if ($LASTEXITCODE -ne 0) { Fail "S3 bucket $Bucket not found. Create it with: aws s3 mb s3://$Bucket --region $Region" }
Ok "bucket $Bucket exists"

aws ecr describe-images --repository-name $Name --image-ids imageTag=latest --region $Region > $null 2>&1
if ($LASTEXITCODE -ne 0) { Fail "No '${Name}:latest' image in ECR. Build and push it first (see README)." }
Ok "ECR image ${Name}:latest exists"

if (-not $Arch) {
    $localArch = docker image inspect "${Name}:latest" --format "{{.Architecture}}" 2>$null
    if ($LASTEXITCODE -eq 0 -and $localArch -match "arm64") { $Arch = "ARM64" } else { $Arch = "X86_64" }
}
Ok "CPU architecture $Arch"

# ---------------------------------------------------------------------------
Step "CloudWatch log group"
$LogGroup = "/ecs/$Name"
$found = aws logs describe-log-groups --log-group-name-prefix $LogGroup --region $Region --query "logGroups[?logGroupName=='$LogGroup'].logGroupName" --output text
if ($found -and $found.Trim() -eq $LogGroup) {
    Ok "log group $LogGroup already exists"
} else {
    aws logs create-log-group --log-group-name $LogGroup --region $Region
    if ($LASTEXITCODE -ne 0) { Fail "could not create log group $LogGroup" }
    Ok "created log group $LogGroup"
}
aws logs put-retention-policy --log-group-name $LogGroup --retention-in-days 30 --region $Region
Ok "logs kept for 30 days"

# ---------------------------------------------------------------------------
Step "IAM roles"
$script:createdRole = $false
$trust = Join-Path $here "ecs-tasks-trust.json"

$ExecRole = "$Name-execution-role"
Ensure-Role $ExecRole $trust
aws iam attach-role-policy --role-name $ExecRole --policy-arn "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
if ($LASTEXITCODE -ne 0) { Fail "could not attach execution policy to $ExecRole" }
Ok "$ExecRole can pull from ECR and write logs"

$TaskRole = "$Name-task-role"
Ensure-Role $TaskRole $trust
$policyFile = Render "task-role-policy.json" @{ BUCKET = $Bucket }
aws iam put-role-policy --role-name $TaskRole --policy-name "$Name-s3-access" --policy-document "file://$policyFile"
if ($LASTEXITCODE -ne 0) { Fail "could not attach S3 policy to $TaskRole" }
Ok "$TaskRole can read/write s3://$Bucket only"

if ($script:createdRole) {
    Write-Host "   waiting 15s for new IAM roles to become active..."
    Start-Sleep -Seconds 15
}

# ---------------------------------------------------------------------------
Step "ECS cluster"
# First use of ECS in an account needs this role; harmless if it already exists.
aws iam create-service-linked-role --aws-service-name ecs.amazonaws.com > $null 2>&1
aws ecs create-cluster --cluster-name $Name --region $Region > $null
if ($LASTEXITCODE -ne 0) { Fail "could not create ECS cluster $Name (if this is your first ECS use, wait a minute and re-run)" }
Ok "cluster $Name ready"

# ---------------------------------------------------------------------------
Step "ECS task definition"
$tdFile = Render "taskdef.json" @{ NAME = $Name; ACCOUNT = $Account; REGION = $Region; BUCKET = $Bucket; ARCH = $Arch }
$tdArn = aws ecs register-task-definition --cli-input-json "file://$tdFile" --region $Region --query "taskDefinition.taskDefinitionArn" --output text
if ($LASTEXITCODE -ne 0) { Fail "could not register task definition" }
Ok "registered $tdArn"

Write-Host "`nSetup complete. Next: powershell -ExecutionPolicy Bypass -File .\deploy\run.ps1 -Command all" -ForegroundColor Cyan
