$ErrorActionPreference='Stop'
$followupWorkspace=Split-Path -Parent $PSScriptRoot
$followupPython=Join-Path $followupWorkspace '.venv\Scripts\python.exe'
$followupStatus=(& $followupPython (Join-Path $PSScriptRoot 'coding_followup.py') --status) | ConvertFrom-Json
if ($LASTEXITCODE -ne 0) { throw 'Cannot inspect follow-up status.' }
if ($followupStatus.study.runner_alive -eq $true) { Write-Output 'Follow-up already running.'; return }
if ($followupStatus.study.status -eq 'unknown') { throw 'Cannot verify queue liveness; refusing a duplicate launch.' }
if ($followupStatus.study.status -eq 'completed') { Write-Output 'Follow-up completed.'; return }
$followupLogs=Join-Path $followupWorkspace 'outputs\coding-followup\logs'
New-Item -ItemType Directory -Path $followupLogs -Force | Out-Null
$followupStamp=Get-Date -Format 'yyyyMMdd-HHmmss-fff'
$followupLog="outputs/coding-followup/logs/runner-$followupStamp.log"
$followupProcess=Start-Process -FilePath $followupPython -ArgumentList "-u experiments/coding_followup.py --runner-log $followupLog" -WorkingDirectory $followupWorkspace -WindowStyle Hidden -PassThru
[PSCustomObject]@{LauncherPid=$followupProcess.Id;RunnerLog=$followupLog}
