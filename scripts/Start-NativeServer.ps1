param([ValidateSet('clean-q8_0','clean-q4_k_m','base-q8_0')][string]$Variant='clean-q8_0',[int]$Port=11436)
$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
if(Test-Path -LiteralPath (Join-Path $root 'data/quant-process.json')){throw '请先停止本项目 Ollama 量化实验实例，避免争用显存。'}
if(Test-Path -LiteralPath (Join-Path $root 'data/reader-process.json')){throw '请先停止本项目阅读网关，再启动或切换原生模型。'}
if(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue){throw "端口 $Port 已占用，未停止现有服务。"}
$record=Join-Path $root 'data/native-process.json'
if(Test-Path -LiteralPath $record){throw '已有原生服务记录，请先检查或运行 Stop-NativeServer.ps1。'}
$modelPath=Join-Path $root "weights/gguf/$Variant.gguf"
$binary=Join-Path $root 'tools/llama-cuda/llama-server.exe'
if(!(Test-Path -LiteralPath $binary)){throw '请先运行 Install-NativeCuda.ps1，禁止静默退回 CPU。'}
$devices=(& $binary --list-devices 2>&1 | Out-String)
if($devices -notmatch 'CUDA0'){throw '未发现 CUDA0，拒绝启动 CPU 回退实例。'}
$artifacts=Get-Content -LiteralPath (Join-Path $root 'reports/gguf-artifacts.json') -Raw | ConvertFrom-Json
$expected=($artifacts.files | Where-Object name -EQ "$Variant.gguf").sha256
$modelHash=(Get-FileHash -LiteralPath $modelPath).Hash.ToLowerInvariant()
if($modelHash -ne $expected){throw 'GGUF 哈希与已有实验记录不一致。'}
New-Item -ItemType Directory -Path (Join-Path $root 'data'),(Join-Path $root 'logs') -Force | Out-Null
$keyFile=Join-Path $root 'data/native.token'
if(!(Test-Path -LiteralPath $keyFile)){
    [IO.File]::WriteAllText($keyFile,[Convert]::ToHexString([Security.Cryptography.RandomNumberGenerator]::GetBytes(32)))
}
$process=Start-Process -FilePath $binary -WorkingDirectory $root -WindowStyle Hidden -PassThru -ArgumentList @(
    '--model',('"'+$modelPath+'"'),'--host','127.0.0.1','--port',$Port,'--parallel','1','--ctx-size','2048',
    '--gpu-layers','all','--flash-attn','on','--no-context-shift','--threads','4','--n-predict','1024',
    '--api-key-file',('"'+$keyFile+'"'),'--slots') -RedirectStandardOutput (Join-Path $root 'logs/native.out.log') -RedirectStandardError (Join-Path $root 'logs/native.err.log')
@{pid=$process.Id;start_time=$process.StartTime.ToUniversalTime().ToString('o');root=$root;port=$Port;variant=$Variant;
  model=$(if($Variant -like 'base-*'){'base'}else{'clean_2k_seed24'});model_path=$modelPath;model_sha256=$modelHash;
  binary_sha256=(Get-FileHash -LiteralPath $binary).Hash.ToLowerInvariant()} | ConvertTo-Json | Set-Content -LiteralPath $record -Encoding UTF8
for($attempt=0;$attempt -lt 90;$attempt++){
    if($process.HasExited){throw '原生服务启动失败，查看 logs/native.err.log。'}
    try{if((Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 2).status -eq 'ok'){Write-Output "原生服务已就绪：$Variant / 127.0.0.1:$Port";exit 0}}catch{}
    Start-Sleep -Seconds 1
}
throw '原生服务加载超时；进程已记录，请检查日志或运行 Stop-NativeServer.ps1。'
