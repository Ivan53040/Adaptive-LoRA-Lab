$ErrorActionPreference = 'Stop'
$rankWorkspace = Split-Path -Parent $PSScriptRoot
$rankPython = Join-Path $rankWorkspace '.venv\Scripts\python.exe'
$rankStatusScript = Join-Path $PSScriptRoot 'rank_ablation.py'
$rankStatusText = & $rankPython $rankStatusScript --status
if ($LASTEXITCODE -ne 0) { throw 'Could not read the rank-study status.' }
$rankStatus = $rankStatusText | ConvertFrom-Json
if ($rankStatus.study.runner_alive -eq $true) {
    Write-Output 'The rank study is already running; preserved the existing process.'
    return
}
if ($rankStatus.study.status -eq 'unknown') {
    throw 'Cannot verify the previous runner; resolve process access before starting another.'
}
if ($rankStatus.study.status -eq 'completed') {
    Write-Output 'The rank study is already complete; inspect comparison.md.'
    return
}
$rankLogs = Join-Path $rankWorkspace 'outputs\rank-ablation-1k\logs'
New-Item -ItemType Directory -Path $rankLogs -Force | Out-Null
$rankLaunchStamp = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
$rankRelativeLog = "outputs/rank-ablation-1k/logs/runner-$rankLaunchStamp.log"
$rankCombinedLog = Join-Path $rankWorkspace $rankRelativeLog
# Let Windows create an independent hidden console. Python opens its own log;
# redirecting handles here can keep the launching execution session attached.
$rankProcess = Start-Process -FilePath $rankPython -ArgumentList "-u experiments/rank_ablation.py --runner-log $rankRelativeLog" -WorkingDirectory $rankWorkspace -WindowStyle Hidden -PassThru
[PSCustomObject]@{
    LauncherPid = $rankProcess.Id
    RunnerLog = $rankCombinedLog
    ProgressCommand = '.\.venv\Scripts\python.exe experiments\rank_ablation.py --status'
}
