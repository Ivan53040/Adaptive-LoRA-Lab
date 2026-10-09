$ErrorActionPreference = 'Stop'
$protocolWorkspace = Split-Path -Parent $PSScriptRoot
$protocolPython = Join-Path $protocolWorkspace '.venv\Scripts\python.exe'
$protocolScript = Join-Path $PSScriptRoot 'protocol_repair.py'
$protocolStatus = (& $protocolPython $protocolScript --status) | ConvertFrom-Json
if ($LASTEXITCODE -ne 0) { throw 'Could not inspect the protocol-repair status.' }
if ($protocolStatus.pipeline.runner_alive -eq $true) {
    Write-Output 'The protocol-repair queue is already running.'
    return
}
if ($protocolStatus.pipeline.status -eq 'unknown') { throw 'Cannot verify the previous runner.' }
if ($protocolStatus.pipeline.status -eq 'completed') {
    Write-Output 'The repair and complete benchmark have finished; inspect repair-report.md.'
    return
}
$protocolStamp = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
$protocolLog = "outputs/protocol-repair/r16-joint-completion/logs/runner-$protocolStamp.log"
$protocolProcess = Start-Process -FilePath $protocolPython -ArgumentList "-u experiments/protocol_repair.py --runner-log $protocolLog" -WorkingDirectory $protocolWorkspace -WindowStyle Hidden -PassThru
[PSCustomObject]@{ LauncherPid = $protocolProcess.Id; RunnerLog = $protocolLog }
