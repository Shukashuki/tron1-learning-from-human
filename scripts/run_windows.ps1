param(
    [string]$Runtime = 'D:\tron1-isaac\isaac-sim-4.5.0',
    [string]$WorkDir = 'D:\tron1-isaac\project',
    [string]$OutputDir = '',
    [int]$Steps = 240,
    [switch]$Balance,
    [double]$Seconds = 15,
    [double]$InitialPitchDeg = 2,
    [ValidateSet('lqr', 'off')]
    [string]$Controller = 'lqr',
    [switch]$Headless,
    [switch]$FreeBase,
    [switch]$KeepOpen
)
$ErrorActionPreference = 'Stop'
$source = Split-Path $PSScriptRoot -Parent
$python = Join-Path $Runtime 'python.bat'
if (-not (Test-Path $python)) { throw "Isaac Sim runtime not found: $python" }
if (-not (Test-Path (Join-Path $source 'assets\robots\WF_TRON1A\WF_TRON1A.usd'))) {
    throw 'Prepare assets with scripts/prepare_assets.py first.'
}
New-Item -ItemType Directory -Force -Path $WorkDir | Out-Null
# Keep Windows runtime I/O on NTFS. Only copy this project's reproducible inputs.
foreach ($name in @('scripts', 'assets', 'config', 'launch_tron1.cmd', 'launch_balance.cmd', 'launch_mocap.cmd', 'launch_mink.cmd', 'launch_sim2sim.cmd', 'README.md')) {
    $item = Join-Path $source $name
    if ((Test-Path $item) -and ($source -ne $WorkDir)) {
        Copy-Item -Path $item -Destination $WorkDir -Recurse -Force
    }
}
if (-not $OutputDir) {
    $OutputDir = Join-Path 'D:\tron1-isaac\runs' (Get-Date -Format 'yyyyMMdd-HHmmss')
}
if ($Balance) {
    $simArgs = @(
        (Join-Path $WorkDir 'scripts\balance_tron1.py'),
        '--seconds', $Seconds.ToString([System.Globalization.CultureInfo]::InvariantCulture),
        '--initial-pitch-deg', $InitialPitchDeg.ToString([System.Globalization.CultureInfo]::InvariantCulture),
        '--controller', $Controller,
        '--output-dir', $OutputDir
    )
    $reportName = 'balance_report.json'
} else {
    $simArgs = @((Join-Path $WorkDir 'scripts\import_tron1.py'), '--steps', "$Steps", '--output-dir', $OutputDir)
    if ($FreeBase) { $simArgs += '--free-base' }
    $reportName = 'import_report.json'
}
if ($Headless) { $simArgs += '--headless' }
if ($KeepOpen) { $simArgs += '--keep-open' }
$env:OMNI_KIT_ACCEPT_EULA = 'YES'
Push-Location $Runtime
try {
    & $python @simArgs
    if ($LASTEXITCODE -ne 0) { throw "Isaac exited with code $LASTEXITCODE" }
    $report = Get-Content (Join-Path $OutputDir $reportName) -Raw | ConvertFrom-Json
    if ($report.status -ne 'passed') { throw "TRON1 run did not pass: $($report.status) ($reportName)" }
    Write-Host "TRON1 ready. Results: $OutputDir"
} finally {
    Pop-Location
}
