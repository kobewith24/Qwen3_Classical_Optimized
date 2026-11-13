param([string]$Model='clean_2k_seed24',[int]$Port=8010,[double]$TaskTimeoutSeconds=300,[switch]$OpenBrowser,
      [ValidateSet('hf','native')][string]$Backend='native',[int]$NativePort=11436,[switch]$Unconstrained)
$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
$localPythonFile = Join-Path $root 'data/python.path'
$venvPython = Join-Path $root '.venv/Scripts/python.exe'
$pythonCommand = if($env:CLASSICAL_PYTHON){$env:CLASSICAL_PYTHON}
    elseif(Test-Path -LiteralPath $localPythonFile){(Get-Content -LiteralPath $localPythonFile -Raw -Encoding UTF8).Trim()}
    elseif(Test-Path -LiteralPath $venvPython){$venvPython}else{'python'}
$resolvedPython = Get-Command -Name $pythonCommand -CommandType Application -ErrorAction Stop
$pythonExecutable = $resolvedPython.Source
if(Test-Path -LiteralPath (Join-Path $root 'data/quant-process.json')){throw '请先停止本项目 Ollama 量化实验实例。'}
if(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue){throw "端口 $Port 已被占用，未停止现有进程。"}
foreach($part in @('logs','data')){New-Item -ItemType Directory -Path (Join-Path $root $part) -Force | Out-Null}
$env:PYTHONIOENCODING='utf-8';$env:PYTHONDONTWRITEBYTECODE='1';$env:PYTHONUNBUFFERED='1'
$env:HF_HUB_OFFLINE='1';$env:TRANSFORMERS_OFFLINE='1';$env:HF_HUB_DISABLE_TELEMETRY='1'
$env:HF_HOME=Join-Path $root 'cache\huggingface';$env:TORCH_HOME=Join-Path $root 'cache\torch'
$env:TRITON_CACHE_DIR=Join-Path $root 'cache\triton'
$env:TORCHINDUCTOR_CACHE_DIR=Join-Path $root 'cache\inductor'
$env:CUDA_CACHE_PATH=Join-Path $root 'cache\cuda'
$env:TEMP=Join-Path $root 'cache\tmp';$env:TMP=$env:TEMP
New-Item -ItemType Directory -Path $env:TEMP -Force | Out-Null
$env:TOKENIZERS_PARALLELISM='false'
$env:WANDB_DISABLED='true'
$nativeOwned=$false
if($Backend -eq 'native'){
    if(!(Test-Path -LiteralPath (Join-Path $root 'data/native-process.json'))){
        if($Model -ne 'clean_2k_seed24'){throw '自动启动目前仅支持 clean_2k_seed24；其他模型请先显式启动对应 GGUF。'}
        & (Join-Path $PSScriptRoot 'Start-NativeServer.ps1') -Port $NativePort
        if($LASTEXITCODE -ne 0){throw '原生服务启动失败。'}
        $nativeOwned=$true
    }
}elseif(Test-Path -LiteralPath (Join-Path $root 'data/native-process.json')){throw '本项目原生模型仍在运行，请先停止以避免与 HF 争用显存。'}
$readerArgs=@('-m','api.server','--model',$Model,'--port',$Port,'--reader-task-timeout',$TaskTimeoutSeconds.ToString([Globalization.CultureInfo]::InvariantCulture),'--backend',$Backend,'--native-port',$NativePort)
if($Unconstrained){$readerArgs+='--unconstrained'}
$process=Start-Process -FilePath $pythonExecutable -ArgumentList $readerArgs -WorkingDirectory $root -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $root 'logs\reader.out.log') -RedirectStandardError (Join-Path $root 'logs\reader.err.log')
@{pid=$process.Id;start_time=$process.StartTime.ToUniversalTime().ToString('o');root=$root;port=$Port;model=$Model;backend=$Backend;native_owned=$nativeOwned} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $root 'data\reader-process.json') -Encoding UTF8
for($attempt=0;$attempt -lt 120;$attempt++){
    if($process.HasExited){throw '模型服务启动失败，请查看 logs/reader.err.log。'}
    try{$health=Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 2;if($health.status -eq 'ready'){
        Write-Output "研读工具已就绪：http://127.0.0.1:$Port";Write-Output '令牌位于新项目 data/access.token。'
        if($OpenBrowser){Start-Process "http://127.0.0.1:$Port"}
        exit 0
    }}catch{}
    Start-Sleep -Seconds 1
}
throw '模型加载等待超时，进程仍记录在 data/reader-process.json。请查看日志或运行 Stop-Reader.ps1。'
