[CmdletBinding()]
param(
    [string]$Root = 'D:\tron1-isaac',
    [switch]$DownloadOnly
)

# Official standalone distribution; keep this release aligned with the importer.
# This script installs only under Root and does not change drivers, the registry,
# PATH, or other machine-wide settings. Existing runtimes are never overwritten.
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$ArtifactVersion = '4.5.0-rc.36+release.19112.f59b3005.gl'
$ArchiveUrl = 'https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone%404.5.0-rc.36%2Brelease.19112.f59b3005.gl.windows-x86_64.release.zip'
$ExpectedBytes = [int64]7013544567
$ExpectedSha256 = '29171fe52efa1923162c215d42907f7f1363dfcff2b4e25a12428ab6cbd08fc8'
# Recorded from the official HTTPS download on 2026-10-01; this checks repeat
# download integrity and is not a vendor-signed checksum.
$Runtime = Join-Path $Root 'isaac-sim-4.5.0'
$Downloads = Join-Path $Root 'downloads'
$Archive = Join-Path $Downloads 'isaac-sim-4.5.0-windows.zip'
$Partial = "$Archive.partial"
$Staging = "$Runtime.extracting"
$RuntimeFiles = @(
    'python.bat',
    'isaac-sim.bat',
    'kit\kit.exe',
    'kit\python\python.exe',
    'apps\isaacsim.exp.full.kit',
    'exts\isaacsim.simulation_app\isaacsim\simulation_app\simulation_app.py',
    'extscache\isaacsim.asset.importer.urdf-2.3.10+106.4.0.wx64.r.cp310\config\extension.toml'
)

if (Test-Path -LiteralPath $Runtime) {
    $VersionFile = Join-Path $Runtime 'VERSION'
    $PythonBatch = Join-Path $Runtime 'python.bat'
    if ((Test-Path -LiteralPath $VersionFile) -and (Test-Path -LiteralPath $PythonBatch)) {
        $ExistingVersion = (Get-Content -LiteralPath $VersionFile -Raw).Trim()
        $MissingFiles = @($RuntimeFiles | Where-Object { -not (Test-Path -LiteralPath (Join-Path $Runtime $_)) })
        if ($ExistingVersion -eq $ArtifactVersion -and $MissingFiles.Count -eq 0) {
            Write-Host "Existing Isaac runtime preserved: $Runtime"
            Write-Host "To check its bundled Python: & '$PythonBatch' --version"
            return
        }
    }
    throw "Existing path was preserved: $Runtime. Select a different -Root for a new installation."
}
if (Test-Path -LiteralPath $Staging) {
    throw "Previous extraction was preserved: $Staging. Inspect it or choose a different -Root."
}

New-Item -ItemType Directory -Path $Downloads -Force | Out-Null
if (-not (Test-Path -LiteralPath $Archive)) {
    # Windows curl supports restartable downloads without loading 7 GB into RAM.
    & curl.exe --fail --location --retry 5 --continue-at - --output $Partial $ArchiveUrl
    if ($LASTEXITCODE -ne 0) {
        throw "Download failed; partial download retained at $Partial for the next run."
    }
    Move-Item -LiteralPath $Partial -Destination $Archive
}
if ((Get-Item -LiteralPath $Archive).Length -ne $ExpectedBytes) {
    throw "Archive size mismatch. File preserved for inspection: $Archive"
}
$ArchiveHash = (Get-FileHash -LiteralPath $Archive -Algorithm SHA256).Hash.ToLowerInvariant()
if ($ExpectedSha256 -and $ArchiveHash -ne $ExpectedSha256) {
    throw "Archive SHA-256 mismatch. File preserved for inspection: $Archive"
}
Write-Host "Archive verified: $Archive"
Write-Host "SHA-256: $ArchiveHash"
if ($DownloadOnly) { return }

Add-Type -AssemblyName System.IO.Compression.FileSystem
$Zip = [System.IO.Compression.ZipFile]::OpenRead($Archive)
try {
    $Prefix = [System.IO.Path]::GetFullPath($Staging).TrimEnd('\') + '\'
    foreach ($Entry in $Zip.Entries) {
        $Target = [System.IO.Path]::GetFullPath((Join-Path $Staging $Entry.FullName))
        if (-not $Target.StartsWith($Prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "Unsafe archive entry: $($Entry.FullName)"
        }
        if ($Target.Length -ge 260) {
            throw 'The chosen root produces long Windows paths. Select a shorter -Root.'
        }
    }
} finally {
    $Zip.Dispose()
}
Write-Host "Extracting approximately 13.5 GB to $Staging ..."
[System.IO.Compression.ZipFile]::ExtractToDirectory($Archive, $Staging)
$ExtractedVersion = (Get-Content -LiteralPath (Join-Path $Staging 'VERSION') -Raw).Trim()
if ($ExtractedVersion -ne $ArtifactVersion) { throw 'Extracted version mismatch; staging preserved.' }
foreach ($RelativePath in $RuntimeFiles) {
    if (-not (Test-Path -LiteralPath (Join-Path $Staging $RelativePath))) {
        throw "Missing runtime file: $RelativePath. Staging preserved."
    }
}
Move-Item -LiteralPath $Staging -Destination $Runtime
& (Join-Path $Runtime 'python.bat') --version
if ($LASTEXITCODE -ne 0) { throw 'Bundled Python check failed; runtime preserved for diagnosis.' }
Write-Host "Isaac Sim ready: $Runtime"
Write-Host 'Next, launch the repository scripts\run_windows.ps1 to import TRON1.'
