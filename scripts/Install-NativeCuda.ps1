$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
$destination=Join-Path $root 'tools/llama-cuda'
if((Test-Path -LiteralPath $destination) -and (Get-ChildItem -LiteralPath $destination)){throw 'CUDA 目录非空，不覆盖已有工具。'}
$downloads=Join-Path $root 'tools/downloads'
New-Item -ItemType Directory -Path $downloads -Force | Out-Null
$packages=@(
    @{name='llama-b12144-bin-win-cuda-12.4-x64.zip';sha256='fab315401a4784fa9c98c3105bf6e2ede6d3d2dafceb873e9bccd169f0f32cb7'},
    @{name='cudart-llama-bin-win-cuda-12.4-x64.zip';sha256='8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6'}
)
foreach($package in $packages){
    $archive=Join-Path $downloads $package.name
    if(!(Test-Path -LiteralPath $archive)){
        Invoke-WebRequest ('https://github.com/ggml-org/llama.cpp/releases/download/b12144/'+$package.name) -OutFile $archive
    }
    if((Get-FileHash -LiteralPath $archive).Hash.ToLowerInvariant() -ne $package.sha256){throw ('下载文件哈希不匹配：'+$package.name)}
}
New-Item -ItemType Directory -Path $destination -Force | Out-Null
foreach($package in $packages){Expand-Archive -LiteralPath (Join-Path $downloads $package.name) -DestinationPath $destination -Force}
New-Item -ItemType Directory -Path (Join-Path $root 'reports/validation') -Force | Out-Null
$packages | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $root 'reports/validation/cuda-packages.json') -Encoding UTF8
& (Join-Path $destination 'llama-server.exe') --list-devices
if($LASTEXITCODE -ne 0){throw 'CUDA 原生工具探测失败。'}
