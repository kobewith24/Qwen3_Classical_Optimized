param([int]$Port=11435,[switch]$BaseOnly)
$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
$env:OLLAMA_HOST="127.0.0.1:$Port"
$env:OLLAMA_MODELS=Join-Path $root 'assets\ollama'
$template=@'
TEMPLATE """{{ range .Messages }}<|im_start|>{{ .Role }}
{{ .Content }}<|im_end|>
{{ end }}<|im_start|>assistant
"""
PARAMETER temperature 0
PARAMETER num_ctx 2048
PARAMETER stop <|im_end|>
'@
$models=@{ 'classical-base-q8'='base-q8_0.gguf' }
if(!$BaseOnly){$models['classical-clean-q8']='clean-q8_0.gguf';$models['classical-clean-q4']='clean-q4_k_m.gguf'}
foreach($name in $models.Keys){
    $file=Join-Path $root ('weights\gguf\'+$models[$name])
    if(!(Test-Path -LiteralPath $file)){throw "缺少量化文件 $file"}
    $modelfile=Join-Path $root ('deploy\Modelfile.'+$name)
    $content='FROM "'+$file.Replace('\','/')+'"'+"`n`n"+$template+"`n"
    [IO.File]::WriteAllText($modelfile,$content,(New-Object Text.UTF8Encoding($false)))
    & ollama create $name -f $modelfile
    if($LASTEXITCODE -ne 0){throw "模型注册失败：$name"}
}
