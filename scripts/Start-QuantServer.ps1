param([int]$Port=11435)
$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
if((Test-Path -LiteralPath (Join-Path $root 'data/native-process.json')) -or (Test-Path -LiteralPath (Join-Path $root 'data/reader-process.json'))){throw '请先停止本项目阅读/原生服务，避免争用显存。'}
foreach($part in @('logs','data')){New-Item -ItemType Directory -Path (Join-Path $root $part) -Force | Out-Null}
if(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue){throw "端口 $Port 已被占用。"}
$env:OLLAMA_HOST="127.0.0.1:$Port"
$env:OLLAMA_MODELS=Join-Path $root 'assets\ollama'
$env:OLLAMA_NUM_PARALLEL='1'
$env:OLLAMA_MAX_LOADED_MODELS='1'
$env:OLLAMA_KEEP_ALIVE='5m'
$env:OLLAMA_NO_CLOUD='1'
$env:CUDA_CACHE_PATH=Join-Path $root 'cache\cuda'
$env:TEMP=Join-Path $root 'cache\tmp';$env:TMP=$env:TEMP
New-Item -ItemType Directory -Force -Path $env:TEMP | Out-Null
New-Item -ItemType Directory -Force -Path $env:OLLAMA_MODELS | Out-Null
$ollama=(Get-Command ollama -ErrorAction Stop).Source
$process=Start-Process -FilePath $ollama -ArgumentList 'serve' -WorkingDirectory $root -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $root 'logs\quant-server.out.log') -RedirectStandardError (Join-Path $root 'logs\quant-server.err.log')
@{pid=$process.Id;start_time=$process.StartTime.ToUniversalTime().ToString('o');root=$root;port=$Port} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $root 'data\quant-process.json') -Encoding UTF8
for($attempt=0;$attempt -lt 30;$attempt++){
    if($process.HasExited){throw '量化测试服务启动失败，请查看新项目日志。'}
    try{Invoke-RestMethod "http://127.0.0.1:$Port/api/version" -TimeoutSec 1 | Out-Null;Write-Output "独立 Ollama 已就绪：127.0.0.1:$Port；模型目录仅在本项目内。";exit}catch{}
    Start-Sleep -Seconds 1
}
throw '独立 Ollama 启动等待超时。'
