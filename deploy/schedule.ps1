# STRIAX - schedule the Fargate pipeline with Amazon EventBridge Scheduler.
#
# Creates (or updates) two schedules, in New York time (daylight saving handled automatically):
#   striax-daily           Mon-Fri 17:15  -> fetch + predict with the saved model
#   striax-weekly-retrain  Sat     10:00  -> fetch + full retrain + predict
# plus an IAM role  striax-scheduler-role  that may only start the striax task.
#
# Run deploy\setup.ps1 first. Safe to re-run.
#
# Usage (from the repo root):
#   powershell -ExecutionPolicy Bypass -File .\deploy\schedule.ps1            # create/update
#   powershell -ExecutionPolicy Bypass -File .\deploy\schedule.ps1 -Pause     # turn schedules off
#   powershell -ExecutionPolicy Bypass -File .\deploy\schedule.ps1 -Remove    # delete schedules
#   powershell -ExecutionPolicy Bypass -File .\deploy\schedule.ps1 -TestNow   # one-time run in 2 minutes

param(
    [string]$Region = "us-east-1",
    [string]$Name = "striax",
    [string]$TimeZone = "America/New_York",
    [string]$DailyCron = "cron(15 17 ? * MON-FRI *)",
    [string]$RetrainCron = "cron(0 10 ? * SAT *)",
    [switch]$Pause,
    [switch]$Remove,
    [switch]$TestNow
)

$here = Split-Path -Parent $MyInvocation.MyCommand.Path

function Fail($msg) { Write-Host "ERROR: $msg" -ForegroundColor Red; exit 1 }
function Step($msg) { Write-Host "`n== $msg" -ForegroundColor Cyan }
function Ok($msg)   { Write-Host "   $msg" -ForegroundColor Green }

function Write-NoBom($path, $text) {
    [System.IO.File]::WriteAllText($path, $text, (New-Object System.Text.UTF8Encoding($false)))
}

function Render($template, $vars) {
    $text = Get-Content -Raw -Path (Join-Path $here $template)
    foreach ($k in $vars.Keys) { $text = $text.Replace("__${k}__", $vars[$k]) }
    $out = Join-Path $env:TEMP ("striax-" + $template)
    Write-NoBom $out $text
    return $out
}

function Schedule-Exists($schedName) {
    aws scheduler get-schedule --name $schedName --region $Region > $null 2>&1
    return ($LASTEXITCODE -eq 0)
}

$DailyName = "$Name-daily"
$RetrainName = "$Name-weekly-retrain"

# ---------------------------------------------------------------------------
Step "Checking prerequisites"
$Account = aws sts get-caller-identity --query Account --output text
if ($LASTEXITCODE -ne 0 -or -not $Account) { Fail "AWS CLI is not configured. Run 'aws configure' first." }
$Account = $Account.Trim()
Ok "account $Account, region $Region"

if ($Remove) {
    Step "Removing schedules"
    foreach ($s in @($DailyName, $RetrainName)) {
        if (Schedule-Exists $s) {
            aws scheduler delete-schedule --name $s --region $Region
            Ok "deleted $s"
        } else { Ok "$s does not exist" }
    }
    Write-Host "`nSchedules removed. The task can still be run by hand with deploy\run.ps1." -ForegroundColor Cyan
    exit 0
}

aws ecs describe-task-definition --task-definition $Name --region $Region > $null 2>&1
if ($LASTEXITCODE -ne 0) { Fail "Task definition '$Name' not found. Run deploy\setup.ps1 first." }
Ok "task definition $Name exists"

# ---------------------------------------------------------------------------
Step "Scheduler IAM role"
$Role = "$Name-scheduler-role"
$vars = @{ NAME = $Name; ACCOUNT = $Account; REGION = $Region }
aws iam get-role --role-name $Role > $null 2>&1
if ($LASTEXITCODE -eq 0) {
    Ok "role $Role already exists"
} else {
    $trustFile = Render "scheduler-trust.json" $vars
    aws iam create-role --role-name $Role --assume-role-policy-document "file://$trustFile" > $null
    if ($LASTEXITCODE -ne 0) { Fail "could not create role $Role" }
    Ok "created role $Role"
    $newRole = $true
}
$policyFile = Render "scheduler-role-policy.json" $vars
aws iam put-role-policy --role-name $Role --policy-name "$Name-run-task" --policy-document "file://$policyFile"
if ($LASTEXITCODE -ne 0) { Fail "could not attach policy to $Role" }
Ok "$Role may only start the $Name task"
if ($newRole) {
    Write-Host "   waiting 15s for the new role to become active..."
    Start-Sleep -Seconds 15
}

# ---------------------------------------------------------------------------
Step "Default network"
$vpc = aws ec2 describe-vpcs --filters "Name=isDefault,Values=true" --region $Region --query "Vpcs[0].VpcId" --output text
if ($LASTEXITCODE -ne 0 -or -not $vpc -or $vpc.Trim() -eq "None") { Fail "No default VPC in $Region." }
$vpc = $vpc.Trim()
$subnets = aws ec2 describe-subnets --filters "Name=vpc-id,Values=$vpc" "Name=default-for-az,Values=true" --region $Region --query "Subnets[].SubnetId" --output text
$subnetArray = @(($subnets -split "\s+") | Where-Object { $_ })
if ($subnetArray.Count -eq 0) { Fail "No default subnets found in $vpc." }
$sg = (aws ec2 describe-security-groups --filters "Name=vpc-id,Values=$vpc" "Name=group-name,Values=default" --region $Region --query "SecurityGroups[0].GroupId" --output text).Trim()
Ok "vpc $vpc, $($subnetArray.Count) subnets, security group $sg"

# Builds the target file: EventBridge Scheduler calls ECS RunTask with these exact parameters.
# TaskDefinition is the family name, so the newest revision registered by setup.ps1 is always used.
function Write-Target($command, $path) {
    $runTask = [ordered]@{
        Cluster              = $Name
        TaskDefinition       = $Name
        LaunchType           = "FARGATE"
        Count                = 1
        NetworkConfiguration = [ordered]@{
            AwsvpcConfiguration = [ordered]@{
                Subnets        = $subnetArray
                SecurityGroups = @($sg)
                AssignPublicIp = "ENABLED"
            }
        }
        Overrides            = [ordered]@{
            ContainerOverrides = @( [ordered]@{ Name = $Name; Command = @($command) } )
        }
    }
    $target = [ordered]@{
        Arn         = "arn:aws:scheduler:::aws-sdk:ecs:runTask"
        RoleArn     = "arn:aws:iam::${Account}:role/$Role"
        Input       = (ConvertTo-Json -InputObject $runTask -Depth 10 -Compress)
        RetryPolicy = [ordered]@{ MaximumRetryAttempts = 2; MaximumEventAgeInSeconds = 3600 }
    }
    Write-NoBom $path (ConvertTo-Json -InputObject $target -Depth 10)
}

function Upsert-Schedule($schedName, $cron, $command, $description) {
    $targetFile = Join-Path $env:TEMP "striax-target-$schedName.json"
    Write-Target $command $targetFile
    $state = "ENABLED"
    if ($Pause) { $state = "DISABLED" }
    $verb = "create-schedule"
    if (Schedule-Exists $schedName) { $verb = "update-schedule" }
    aws scheduler $verb --name $schedName --description $description `
        --schedule-expression $cron --schedule-expression-timezone $TimeZone `
        --flexible-time-window "Mode=OFF" --state $state `
        --target "file://$targetFile" --region $Region > $null
    if ($LASTEXITCODE -ne 0) { Fail "could not $verb $schedName" }
    Ok "$schedName  $cron  ($TimeZone)  -> '$command'  [$state]"
}

if ($TestNow) {
    Step "One-time test run"
    $testName = "$Name-test-" + (Get-Date).ToUniversalTime().ToString("yyyyMMddHHmmss")
    $when = (Get-Date).ToUniversalTime().AddMinutes(2).ToString("yyyy-MM-ddTHH:mm:ss")
    $targetFile = Join-Path $env:TEMP "striax-target-test.json"
    Write-Target "daily" $targetFile
    aws scheduler create-schedule --name $testName --description "STRIAX one-time scheduler test" `
        --schedule-expression "at($when)" --schedule-expression-timezone "UTC" `
        --flexible-time-window "Mode=OFF" --action-after-completion DELETE `
        --target "file://$targetFile" --region $Region > $null
    if ($LASTEXITCODE -ne 0) { Fail "could not create test schedule" }
    Ok "$testName fires at $when UTC (about 2 minutes), then deletes itself"
    Write-Host "`nIn about 5 minutes, check that it ran:" -ForegroundColor Cyan
    Write-Host "   aws ecs list-tasks --cluster $Name --desired-status STOPPED --region $Region"
    Write-Host "   aws logs tail /ecs/$Name --since 15m --region $Region"
    exit 0
}

# ---------------------------------------------------------------------------
Step "Schedules"
Upsert-Schedule $DailyName $DailyCron "daily" "STRIAX: fetch prices and predict next trading day (weekdays after close)"
Upsert-Schedule $RetrainName $RetrainCron "all" "STRIAX: weekly full retrain + predict"

Step "Next runs"
foreach ($s in @($DailyName, $RetrainName)) {
    aws scheduler get-schedule --name $s --region $Region --query "{Name:Name,Schedule:ScheduleExpression,TimeZone:ScheduleExpressionTimezone,State:State}" --output table
}

if ($Pause) {
    Write-Host "`nSchedules are paused. Re-run without -Pause to turn them back on." -ForegroundColor Yellow
} else {
    Write-Host "`nDone. STRIAX now runs on its own. Check results any time with:" -ForegroundColor Cyan
    Write-Host "   aws logs tail /ecs/$Name --since 1d --region $Region"
    Write-Host "   aws s3 cp s3://$Name-$Account/out/latest.json -"
}
