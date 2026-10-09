$ErrorActionPreference = 'Stop'
$fairWorkspace = Split-Path -Parent $PSScriptRoot
$fairPython = Join-Path $fairWorkspace '.venv\Scripts\python.exe'
$fairScript = Join-Path $PSScriptRoot 'fair_rank.py'
$fairStatus = (& $fairPython $fairScript --status) | ConvertFrom-Json
if ($LASTEXITCODE -ne 0) { throw 'Could not inspect the fair-rank queue.' }
if ($fairStatus.study.runner_alive -eq $true) { Write-Output 'Fair rank study already running.'; return }
if ($fairStatus.study.status -eq 'unknown') { throw 'Cannot verify the previous runner.' }
if ($fairStatus.study.status -eq 'completed') { Write-Output 'Fair rank study complete; inspect comparison.md.'; return }
$fairStamp = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
$fairLog = "outputs/fair-rank-completion-1k/logs/runner-$fairStamp.log"
$fairProcess = Start-Process -FilePath $fairPython -ArgumentList "-u experiments/fair_rank.py --runner-log $fairLog" -WorkingDirectory $fairWorkspace -WindowStyle Hidden -PassThru
[PSCustomObject]@{LauncherPid=$fairProcess.Id;RunnerLog=$fairLog}
