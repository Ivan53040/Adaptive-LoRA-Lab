$ErrorActionPreference='Stop'
$codingWorkspace=Split-Path -Parent $PSScriptRoot
$codingPython=Join-Path $codingWorkspace '.venv\Scripts\python.exe'
$codingStatus=(& $codingPython (Join-Path $PSScriptRoot 'coding_study.py') --status) | ConvertFrom-Json
if ($LASTEXITCODE -ne 0) { throw 'Cannot inspect coding study status.' }
if ($codingStatus.study.runner_alive -eq $true) { Write-Output 'Coding study already running.'; return }
if ($codingStatus.study.status -eq 'unknown') { throw 'Previous coding runner cannot be verified.' }
if ($codingStatus.study.status -eq 'completed') { Write-Output 'Coding study completed; inspect comparison.md.'; return }
$codingOutput=(Get-Content -LiteralPath (Join-Path $codingWorkspace 'configs\coding\mbpp_r8.json') -Raw | ConvertFrom-Json).output_dir
$codingLogs=Join-Path $codingWorkspace "$codingOutput/logs"
New-Item -ItemType Directory -Path $codingLogs -Force | Out-Null
$codingStamp=Get-Date -Format 'yyyyMMdd-HHmmss-fff'
$codingLog="$codingOutput/logs/runner-$codingStamp.log"
$codingProcess=Start-Process -FilePath $codingPython -ArgumentList "-u experiments/coding_study.py --runner-log $codingLog" -WorkingDirectory $codingWorkspace -WindowStyle Hidden -PassThru
[PSCustomObject]@{LauncherPid=$codingProcess.Id;RunnerLog=$codingLog}
