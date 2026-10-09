$ErrorActionPreference='Stop'
$expansionWorkspace=Split-Path -Parent $PSScriptRoot
$expansionPython=Join-Path $expansionWorkspace '.venv\Scripts\python.exe'
$expansionStatus=(& $expansionPython (Join-Path $PSScriptRoot 'coding_expansion.py') --status) | ConvertFrom-Json
if ($LASTEXITCODE -ne 0) { throw 'Cannot inspect expansion status.' }
if ($expansionStatus.study.runner_alive -eq $true) { Write-Output 'Expansion queue already running.'; return }
if ($expansionStatus.study.status -eq 'unknown') { throw 'Cannot verify queue liveness; refusing a duplicate launch.' }
if ($expansionStatus.study.status -eq 'completed') { Write-Output 'Expansion completed.'; return }
$expansionLogs=Join-Path $expansionWorkspace 'outputs\coding-expansion-3k\logs'
New-Item -ItemType Directory -Path $expansionLogs -Force | Out-Null
$expansionStamp=Get-Date -Format 'yyyyMMdd-HHmmss-fff'
$expansionLog="outputs/coding-expansion-3k/logs/runner-$expansionStamp.log"
$expansionProcess=Start-Process -FilePath $expansionPython -ArgumentList "-u experiments/coding_expansion.py --runner-log $expansionLog" -WorkingDirectory $expansionWorkspace -WindowStyle Hidden -PassThru
[PSCustomObject]@{LauncherPid=$expansionProcess.Id;RunnerLog=$expansionLog}
