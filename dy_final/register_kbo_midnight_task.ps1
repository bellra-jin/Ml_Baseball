param(
    [string]$PythonExe = "C:\Users\Admin\Documents\GitHub\Ml_Baseball\.venv\Scripts\python.exe",
    [string]$TaskName = "KBO 2026 Midnight Update",
    [string]$DataDir = "C:\Users\Admin\Documents\GitHub\Ml_Baseball\data",
    [int]$Year = 2026
)

$RunnerPath = Join-Path $PSScriptRoot "run_kbo_midnight_update.ps1"
$Arguments = "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$RunnerPath`" -PythonExe `"$PythonExe`" -Year $Year -DataDir `"$DataDir`""

$Action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $Arguments
$Trigger = New-ScheduledTaskTrigger -Daily -At 12:00AM
$Settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew

Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings -Description "Update KBO 2026 regular-season CSV files from koreabaseball.com at midnight KST." -Force
Write-Host "Registered task: $TaskName"
Write-Host "Runner: $RunnerPath"
Write-Host "Logs: $PSScriptRoot\logs"
