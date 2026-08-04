$ErrorActionPreference = "Stop"

$timeoutSeconds = 600
$pollSeconds = 2
$startedAt = Get-Date

$codexProcessPatterns = @(
    "^Codex$",
    "^codex$",
    "^codex-command-runner"
)

$codexProfileRoot = Join-Path $env:APPDATA "Codex\web\Codex"
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$backupRoot = Join-Path $env:APPDATA ("Codex\backup\html-reset-" + $timestamp)
$resetTargets = @(
    "Default",
    "GPUPersistentCache",
    "GrShaderCache",
    "ShaderCache",
    "Local State",
    "BrowserMetrics",
    "BrowserMetrics-spare.pma",
    "lockfile"
)

function Get-CodexProcesses {
    Get-Process -ErrorAction SilentlyContinue | Where-Object {
        foreach ($pattern in $codexProcessPatterns) {
            if ($_.ProcessName -match $pattern) {
                return $true
            }
        }
        return $false
    }
}

while ($true) {
    $running = @(Get-CodexProcesses)
    if ($running.Count -eq 0) {
        break
    }

    $elapsedSeconds = [int]((Get-Date) - $startedAt).TotalSeconds
    if ($elapsedSeconds -ge $timeoutSeconds) {
        throw "Timed out waiting for Codex to close."
    }

    Start-Sleep -Seconds $pollSeconds
}

if (-not (Test-Path -LiteralPath $codexProfileRoot)) {
    throw "Codex profile directory not found: $codexProfileRoot"
}

New-Item -ItemType Directory -Path $backupRoot -Force | Out-Null

foreach ($name in $resetTargets) {
    $source = Join-Path $codexProfileRoot $name
    if (-not (Test-Path -LiteralPath $source)) {
        continue
    }

    $destination = Join-Path $backupRoot $name
    Move-Item -LiteralPath $source -Destination $destination -Force
}

$codexExe = (Get-Command codex.exe -ErrorAction Stop | Select-Object -First 1).Source
Start-Process -FilePath $codexExe
