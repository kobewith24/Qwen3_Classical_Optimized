$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
$path=Join-Path $root 'data\reader-process.json'
if(!(Test-Path -LiteralPath $path)){Write-Output '没有本项目的阅读服务启动记录。';exit}
$state=Get-Content -LiteralPath $path -Raw | ConvertFrom-Json
if([IO.Path]::GetFullPath($state.root) -ne [IO.Path]::GetFullPath($root)){throw '启动记录不属于此目录。'}
$process=Get-Process -Id $state.pid -ErrorAction SilentlyContinue
if($process){
    $expected=([DateTimeOffset]$state.start_time).UtcDateTime
    if([Math]::Abs(($process.StartTime.ToUniversalTime()-$expected).TotalSeconds)>2){throw 'PID 已被复用，拒绝停止。'}
    Stop-Process -Id $process.Id
}
Remove-Item -LiteralPath $path
if($state.native_owned){& (Join-Path $PSScriptRoot 'Stop-NativeServer.ps1')}
Write-Output '已停止本项目记录的阅读服务。'
