param(
    [string]$PythonExe = "C:\Users\Admin\Documents\GitHub\Ml_Baseball\.venv\Scripts\python.exe",
    [int]$Year = 2026,
    [string]$DataDir = "C:\Users\Admin\Documents\GitHub\Ml_Baseball\data",
    [switch]$IncludeProfiles
)

$ErrorActionPreference = "Stop"

$ScriptRootPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$CrawlerPath = Join-Path $ScriptRootPath "kbo_realtime_crawler.py"
$PipelineOutputRunner = Join-Path $ScriptRootPath "run_kbo_pipeline_outputs.py"
$VisualizationPath = Join-Path $ScriptRootPath "generate_kbo_visualizations.py"
$DashboardPath = Join-Path $ScriptRootPath "generate_kbo_master_dashboard.py"
$LogDir = Join-Path $ScriptRootPath "logs"
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogFile = Join-Path $LogDir "kbo_midnight_update_$Stamp.log"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Write-Log {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    $line | Tee-Object -FilePath $LogFile -Append
}

try {
    Write-Log "KBO midnight update started."
    Write-Log "PythonExe: $PythonExe"
    Write-Log "CrawlerPath: $CrawlerPath"
    Write-Log "PipelineOutputRunner: $PipelineOutputRunner"
    Write-Log "VisualizationPath: $VisualizationPath"
    Write-Log "DashboardPath: $DashboardPath"
    Write-Log "Year: $Year"
    Write-Log "DataDir: $DataDir"

    if (-not (Test-Path -LiteralPath $PythonExe)) {
        throw "Python executable not found: $PythonExe"
    }
    if (-not (Test-Path -LiteralPath $CrawlerPath)) {
        throw "Crawler script not found: $CrawlerPath"
    }

    $crawlerArgs = @(
        $CrawlerPath,
        "--year", $Year,
        "--data-dir", $DataDir
    )

    if ($IncludeProfiles) {
        $crawlerArgs += "--profiles"
    }

    & $PythonExe @crawlerArgs *>&1 | Tee-Object -FilePath $LogFile -Append

    if ($LASTEXITCODE -ne 0) {
        throw "Crawler failed with exit code $LASTEXITCODE"
    }

    if (Test-Path -LiteralPath $PipelineOutputRunner) {
        Write-Log "Refreshing pipeline prediction CSV files."
        & $PythonExe $PipelineOutputRunner --data-dir $DataDir *>&1 | Tee-Object -FilePath $LogFile -Append
        if ($LASTEXITCODE -ne 0) {
            throw "Pipeline output refresh failed with exit code $LASTEXITCODE"
        }
    }

    if (Test-Path -LiteralPath $VisualizationPath) {
        Write-Log "Regenerating visualization PNG files."
        & $PythonExe $VisualizationPath *>&1 | Tee-Object -FilePath $LogFile -Append
        if ($LASTEXITCODE -ne 0) {
            throw "Visualization generation failed with exit code $LASTEXITCODE"
        }
    }

    if (Test-Path -LiteralPath $DashboardPath) {
        Write-Log "Regenerating dashboard HTML."
        & $PythonExe $DashboardPath *>&1 | Tee-Object -FilePath $LogFile -Append
        if ($LASTEXITCODE -ne 0) {
            throw "Dashboard generation failed with exit code $LASTEXITCODE"
        }
    }

    Write-Log "KBO midnight update finished."
}
catch {
    Write-Log "KBO midnight update failed: $($_.Exception.Message)"
    exit 1
}
