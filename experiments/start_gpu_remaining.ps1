$ErrorActionPreference='Stop'
$remainingWorkspace=Split-Path -Parent $PSScriptRoot
$remainingPython=Join-Path $remainingWorkspace '.venv\Scripts\python.exe'
$remainingStatus=(& $remainingPython (Join-Path $PSScriptRoot 'gpu_remaining.py') --status) | ConvertFrom-Json
if ($LASTEXITCODE -ne 0) { throw 'Cannot inspect remaining-study status.' }
if ($remainingStatus.study.runner_alive -eq $true) { Write-Output 'Remaining-study queue already running.';return }
if ($remainingStatus.study.status -eq 'unknown') { throw 'Cannot verify queue liveness; refusing duplicate launch.' }
if ($remainingStatus.study.status -eq 'completed') { Write-Output 'Remaining-study queue completed.';return }
$remainingLogs=Join-Path $remainingWorkspace 'outputs\gpu-remaining-20261007\logs'
New-Item -ItemType Directory -Path $remainingLogs -Force | Out-Null
$remainingStamp=Get-Date -Format 'yyyyMMdd-HHmmss-fff'
$remainingLog="outputs/gpu-remaining-20261007/logs/runner-$remainingStamp.log"
$remainingProcess=Start-Process -FilePath $remainingPython -ArgumentList "-u experiments/gpu_remaining.py --runner-log $remainingLog" -WorkingDirectory $remainingWorkspace -WindowStyle Hidden -PassThru
[PSCustomObject]@{LauncherPid=$remainingProcess.Id;RunnerLog=$remainingLog}
