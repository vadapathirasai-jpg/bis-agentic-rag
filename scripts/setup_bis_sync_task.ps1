<#
.SYNOPSIS
    Registers the BIS Sahayak incremental synchronization job in Windows Task Scheduler.

.DESCRIPTION
    Creates or updates a scheduled task named 'BIS-Sahayak-Source-Sync' that runs
    once every 24 hours using the project's dedicated Python virtual environment.
    Prevents concurrent executions, starts in the project directory, and integrates
    with the existing run_sync.py file-locking mechanism.

.PARAMETER TaskName
    Name of the scheduled task. Defaults to 'BIS-Sahayak-Source-Sync'.

.PARAMETER ProjectDir
    Root directory of the BIS Sahayak project. Defaults to 'D:\Project\BIS Sahayak'.

.PARAMETER DailyTime
    Time of day to run synchronization (24-hour format HH:mm). Defaults to '03:00' (3:00 AM).

.PARAMETER DryRun
    If specified, validates paths and outputs the task configuration without registering.

.PARAMETER Force
    Overwrites any existing scheduled task with the same name.

.EXAMPLE
    .\scripts\setup_bis_sync_task.ps1 -DryRun
    Validates task configuration and prints details without creating the task.

.EXAMPLE
    .\scripts\setup_bis_sync_task.ps1
    Registers the task to run daily at 3:00 AM.
#>

[CmdletBinding()]
param(
    [string]$TaskName = "BIS-Sahayak-Source-Sync",
    [string]$ProjectDir = "D:\Project\BIS Sahayak",
    [string]$DailyTime = "03:00",
    [switch]$DryRun,
    [switch]$Force
)

$ErrorActionPreference = "Stop"

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host " BIS Sahayak — Task Scheduler Setup (Step 4B)" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

# 1. Resolve & Validate Paths
if (-not (Test-Path -Path $ProjectDir)) {
    Write-Error "Project directory does not exist: $ProjectDir"
    exit 1
}

$ResolvedProjectDir = (Resolve-Path $ProjectDir).Path
$PythonExe = Join-Path $ResolvedProjectDir "backend\venv\Scripts\python.exe"
$RunnerScript = Join-Path $ResolvedProjectDir "backend\app\ingestion\run_sync.py"
$LogFile = Join-Path $ResolvedProjectDir "backend\data\sync_task.log"

if (-not (Test-Path -Path $PythonExe)) {
    Write-Error "Python virtualenv executable not found at: $PythonExe"
    exit 1
}

if (-not (Test-Path -Path $RunnerScript)) {
    Write-Error "Runner script not found at: $RunnerScript"
    exit 1
}

# 2. Check Admin Privileges
$CurrentPrincipal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
$IsAdmin = $CurrentPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
$CurrentUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name

Write-Host "`nEnvironment Verification:" -ForegroundColor Green
Write-Host "  - Project Directory : $ResolvedProjectDir"
Write-Host "  - Python Binary     : $PythonExe"
Write-Host "  - Runner Script     : $RunnerScript"
Write-Host "  - Log Target        : $LogFile"
Write-Host "  - Current User      : $CurrentUser"
Write-Host "  - Elevated (Admin)  : $IsAdmin"

# 3. Configure Task Components
$TriggerTime = [datetime]::ParseExact($DailyTime, "HH:mm", $null)
$Trigger = New-ScheduledTaskTrigger -Daily -At $TriggerTime.ToString("hh:mmtt")

# Action executes python -m app.ingestion.run_sync starting in the project directory
$Arguments = "-m app.ingestion.run_sync"
$Action = New-ScheduledTaskAction -Execute $PythonExe -Argument $Arguments -WorkingDirectory $ResolvedProjectDir

# Settings: Prevent concurrent instances, stop if running too long, allow start on battery/AC
$Settings = New-ScheduledTaskSettingsSet `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2) `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable

$Description = "BIS Sahayak incremental source synchronization job. Checks official BIS sources for content updates once every 24 hours and idempotently updates Qdrant."

Write-Host "`nScheduled Task Specification:" -ForegroundColor Yellow
Write-Host "  - Task Name         : $TaskName"
Write-Host "  - Schedule          : Daily at $($TriggerTime.ToString('HH:mm')) (every 24 hours)"
Write-Host "  - Concurrency Rule  : IgnoreNew (never launch parallel sync jobs)"
Write-Host "  - Working Directory : $ResolvedProjectDir"
Write-Host "  - Program           : $PythonExe"
Write-Host "  - Arguments         : $Arguments"
Write-Host "  - Max Runtime Limit : 2 hours"
Write-Host "  - Safety Lock       : backend\data\.sync.lock (kernel file lock)"

# 4. Dry-Run Check
if ($DryRun) {
    Write-Host "`n[DRY-RUN] Task specification is valid. No changes made to Windows Task Scheduler." -ForegroundColor Green
    Write-Host "Run without -DryRun to register this task.`n"
    exit 0
}

# 5. Check for Existing Task
$ExistingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue

if ($ExistingTask -and -not $Force) {
    Write-Host "`n[NOTICE] Task '$TaskName' already exists. Use -Force to replace it." -ForegroundColor Yellow
    exit 0
}

# 6. Register Task
try {
    Register-ScheduledTask `
        -TaskName $TaskName `
        -Action $Action `
        -Trigger $Trigger `
        -Settings $Settings `
        -Description $Description `
        -Force | Out-Null

    Write-Host "`n[SUCCESS] Scheduled task '$TaskName' has been registered successfully!" -ForegroundColor Green
    Write-Host "Next run scheduled for: $($TriggerTime.ToString('yyyy-MM-dd HH:mm'))`n"
}
catch {
    Write-Error "Failed to register scheduled task: $_"
    if (-not $IsAdmin) {
        Write-Host "[HINT] Creating or updating scheduled tasks may require running PowerShell as Administrator." -ForegroundColor Yellow
    }
    exit 1
}
