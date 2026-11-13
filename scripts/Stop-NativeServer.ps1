$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
$record=Join-Path $root 'data/native-process.json'
if(!(Test-Path -LiteralPath $record)){Write-Output '没有本项目原生服务记录。';exit}
$state=Get-Content -LiteralPath $record -Raw | ConvertFrom-Json
if([IO.Path]::GetFullPath($state.root) -ne [IO.Path]::GetFullPath($root)){throw '服务记录目录不匹配。'}
$process=Get-Process -Id $state.pid -ErrorAction SilentlyContinue
if($process){
    if([Math]::Abs(($process.StartTime.ToUniversalTime()-([DateTimeOffset]$state.start_time).UtcDateTime).TotalSeconds)>2){throw 'PID 已复用，拒绝停止。'}
    Stop-Process -Id $process.Id
}
Remove-Item -LiteralPath $record
Write-Output '已停止本项目记录的原生服务。'
