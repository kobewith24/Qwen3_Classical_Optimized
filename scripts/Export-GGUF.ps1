# Reproduce the seed24 release artifacts; refuse to overwrite recorded GGUF files.
$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root
$target=Join-Path $root 'weights\gguf'
foreach($name in @('base-q8_0.gguf','clean-f16.gguf','clean-q8_0.gguf','clean-q4_k_m.gguf')){
    if(Test-Path -LiteralPath (Join-Path $target $name)){throw "产物 $name 已存在，未覆盖。当前版本可直接用于注册和评测。"}
}
$adapter=Join-Path $root 'weights\clean_2k_seed24\adapter_model.safetensors'
$quantizer=Join-Path $root 'tools\llama-bin\llama-quantize.exe'
if(!(Test-Path -LiteralPath $adapter)){throw '请先完成正式 seed24 训练。'}
if(!(Test-Path -LiteralPath $quantizer)){throw '缺少本地 llama-quantize.exe；工具版本见 logs/quant-clean-q8.log。'}
New-Item -ItemType Directory -Path $target -Force | Out-Null
$merged=Join-Path $root 'weights\merged_clean_2k_seed24'
if(!(Test-Path -LiteralPath $merged)){
    & '.\scripts\Run-Python.ps1' -PythonArgs @('-m','classical.merge','--name','clean_2k_seed24','--shard-mib','128')
    if($LASTEXITCODE -ne 0){throw '完整精度合并失败。'}
}else{
    $report=Get-Content -LiteralPath (Join-Path $root 'reports\merge-clean_2k_seed24.json') -Raw | ConvertFrom-Json
    if($report.adapter_sha256 -ne (Get-FileHash -LiteralPath $adapter -Algorithm SHA256).Hash){throw '现有合并模型不匹配当前 adapter。'}
}
& '.\scripts\Run-Python.ps1' -PythonArgs @('scripts/validate_merge.py')
if($LASTEXITCODE -ne 0){throw '独立合并数值核对失败。'}
$previousPythonPath=$env:PYTHONPATH
try{
    $env:PYTHONPATH=Join-Path $root 'tools\llama.cpp\gguf-py'
    & '.\scripts\Run-Python.ps1' -PythonArgs @('tools/llama.cpp/convert_hf_to_gguf.py','assets/base_model','--outfile','weights/gguf/base-q8_0.gguf','--outtype','q8_0')
    if($LASTEXITCODE -ne 0){throw '基座 GGUF 转换失败。'}
    & '.\scripts\Run-Python.ps1' -PythonArgs @('tools/llama.cpp/convert_hf_to_gguf.py','weights/merged_clean_2k_seed24','--outfile','weights/gguf/clean-f16.gguf','--outtype','f16')
    if($LASTEXITCODE -ne 0){throw '新模型 FP16 GGUF 转换失败。'}
}finally{$env:PYTHONPATH=$previousPythonPath}
& $quantizer 'weights/gguf/clean-f16.gguf' 'weights/gguf/clean-q8_0.gguf' 'Q8_0' '2'
if($LASTEXITCODE -ne 0){throw 'Q8 量化失败。'}
& $quantizer 'weights/gguf/clean-f16.gguf' 'weights/gguf/clean-q4_k_m.gguf' 'Q4_K_M' '2'
if($LASTEXITCODE -ne 0){throw 'Q4 量化失败。'}
& '.\scripts\Run-Python.ps1' -PythonArgs @('scripts/record_quant_artifacts.py')
if($LASTEXITCODE -ne 0){throw 'GGUF 文件校验失败。'}
