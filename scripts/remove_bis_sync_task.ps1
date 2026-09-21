<#
.SYNOPSIS
    Removes the BIS Sahayak scheduled task from Windows Task Scheduler.

.DESCRIPTION
    Safely unregisters only the 'BIS-Sahayak-Source-Sync' scheduled task.
    Does not touch any other system tasks, files, or services.

.PARAMETER TaskName
    Name of the task to remove. Defaults to 'BIS-Sahayak-Source-Sync'.

.EXAMPLE
    .\scripts\remove_bis_sync_task.ps1
#>

[CmdletBinding()]
param(
    [string]$TaskName = "BIS-Sahayak-Source-Sync"
)

$ErrorActionPreference = "Stop"

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host " BIS Sahayak — Task Scheduler Removal (Step 4B)" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

$ExistingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue

if (-not $ExistingTask) {
    Write-Host "`n[INFO] Task '$TaskName' is not registered in Windows Task Scheduler. Nothing to remove.`n" -ForegroundColor Yellow
    exit 0
}

try {
    # Stop the task if currently running
    if ($ExistingTask.State -eq "Running") {
        Write-Host "Stopping running instance of '$TaskName'..." -ForegroundColor Yellow
        Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 1
    }

    # Unregister task
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false | Out-Null
    Write-Host "`n[SUCCESS] Task '$TaskName' has been removed from Windows Task Scheduler.`n" -ForegroundColor Green
}
catch {
    Write-Error "Failed to remove scheduled task '$TaskName': $_"
    exit 1
}
