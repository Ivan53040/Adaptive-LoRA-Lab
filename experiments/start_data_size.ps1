$ErrorActionPreference='Stop'
$dataStudyWorkspace=Split-Path -Parent $PSScriptRoot
$dataStudyPython=Join-Path $dataStudyWorkspace '.venv\Scripts\python.exe'
$dataStudyScript=Join-Path $PSScriptRoot 'data_size.py'
$dataStudyStatus=(& $dataStudyPython $dataStudyScript --status) | ConvertFrom-Json
if ($LASTEXITCODE -ne 0) { throw 'Could not inspect the data-size study.' }
if ($dataStudyStatus.study.runner_alive -eq $true) { Write-Output 'Data-size study already running.'; return }
if ($dataStudyStatus.study.status -eq 'unknown') { throw 'Cannot verify the previous runner.' }
if ($dataStudyStatus.study.status -eq 'completed') { Write-Output 'Data-size study completed; inspect comparison.md.'; return }
$dataStudyStamp=Get-Date -Format 'yyyyMMdd-HHmmss-fff'
$dataStudyLog="outputs/data-size-r8/logs/runner-$dataStudyStamp.log"
$dataStudyProcess=Start-Process -FilePath $dataStudyPython -ArgumentList "-u experiments/data_size.py --runner-log $dataStudyLog" -WorkingDirectory $dataStudyWorkspace -WindowStyle Hidden -PassThru
[PSCustomObject]@{LauncherPid=$dataStudyProcess.Id;RunnerLog=$dataStudyLog}
