$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
$path=Join-Path $root 'data\quant-process.json'
if(!(Test-Path -LiteralPath $path)){exit}
$state=Get-Content -LiteralPath $path -Raw | ConvertFrom-Json
if([IO.Path]::GetFullPath($state.root) -ne [IO.Path]::GetFullPath($root)){throw '启动记录与项目不符。'}
$process=Get-Process -Id $state.pid -ErrorAction SilentlyContinue
if($process){
    if([Math]::Abs(($process.StartTime.ToUniversalTime()-([DateTimeOffset]$state.start_time).UtcDateTime).TotalSeconds)>2){throw 'PID 已复用。'}
    # Unload only this isolated registry before stopping its parent process.
    $url='http://127.0.0.1:'+([int]$state.port)
    $resident=Invoke-RestMethod ($url+'/api/ps') -TimeoutSec 5
    foreach($model in $resident.models){
        $body=@{model=$model.name;keep_alive=0;stream=$false} | ConvertTo-Json
        Invoke-RestMethod ($url+'/api/generate') -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 30 | Out-Null
    }
    Stop-Process -Id $process.Id
}
Remove-Item -LiteralPath $path
Write-Output '已停止本项目独立的量化测试服务。'
