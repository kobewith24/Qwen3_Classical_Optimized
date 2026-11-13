param([Parameter(ValueFromRemainingArguments=$true)][string[]]$PythonArgs)
$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
$localPythonFile = Join-Path $root 'data/python.path'
$venvPython = Join-Path $root '.venv/Scripts/python.exe'
$pythonCommand = if($env:CLASSICAL_PYTHON){$env:CLASSICAL_PYTHON}
    elseif(Test-Path -LiteralPath $localPythonFile){(Get-Content -LiteralPath $localPythonFile -Raw -Encoding UTF8).Trim()}
    elseif(Test-Path -LiteralPath $venvPython){$venvPython}else{'python'}
$resolvedPython = Get-Command -Name $pythonCommand -CommandType Application -ErrorAction Stop
$pythonExecutable = $resolvedPython.Source
Set-Location -LiteralPath $root
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONUNBUFFERED='1'
$env:HF_HUB_DISABLE_TELEMETRY='1'
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
$env:HF_HOME=Join-Path $root 'cache\huggingface'
$env:TORCH_HOME=Join-Path $root 'cache\torch'
$env:TRITON_CACHE_DIR=Join-Path $root 'cache\triton'
$env:TORCHINDUCTOR_CACHE_DIR=Join-Path $root 'cache\inductor'
$env:CUDA_CACHE_PATH=Join-Path $root 'cache\cuda'
$env:TEMP=Join-Path $root 'cache\tmp'
$env:TMP=$env:TEMP
New-Item -ItemType Directory -Path $env:TEMP -Force | Out-Null
$env:TOKENIZERS_PARALLELISM='false'
$env:WANDB_DISABLED='true'
& $pythonExecutable @PythonArgs
exit $LASTEXITCODE
