# Qwen3 文言文微调与本地研读

面向古籍辅助阅读的本地工具：用 Qwen3-4B 完成原文保真断句和古今互译，支持导入 TXT、逐段校对、保存结果、中断后继续处理及导出。模型训练、推理与阅读数据均可在本机完成。

技术栈：**PyTorch · Transformers · PEFT / QLoRA · FastAPI · SQLite**。已验证设备为 RTX 4060 Laptop 8GB，默认使用 `clean_2k_seed24` 的 Q8 部署产物。

![古文研读与校对结果恢复](reports/validation/reader-resume.png)

## 主要能力

- **原文约束断句**：把输入原文编译成解码语法，只开放标点插入，并独立检查输出完整性，减少改字、漏字和续写。
- **领域微调与受控评测**：沿用 gujilab 语料，按典籍划分、去重和检查训练/测试重合；仅监督答案 token，完成两次固定种子的 QLoRA 实验。
- **本地量化推理**：从完整精度基座与 LoRA 分片合并，导出同 checkpoint 的 Q8/Q4 并比较；原生 CUDA 后端提供断句约束，HF NF4 后端可用于诊断。
- **可继续的阅读流程**：SQLite 保存原文、已完成段及人工修订；刷新或服务重启后可以恢复，未完成段重新生成；revision 检查防止旧页面覆盖新校对。
- **单卡任务管理**：实际生成并发 1，等待容量 4；提供进度、取消、超时和 OpenAI 兼容的非流式/SSE 聊天接口，使用真实 token usage。

```mermaid
flowchart LR
    D[gujilab 语料] --> P[典籍划分与重合检查]
    P --> T[答案 token 监督 / QLoRA]
    T --> A[LoRA adapter]
    A --> M[完整精度分片合并]
    M --> Q[Q8 / Q4]
    Q --> N[本机 CUDA 推理]
    U[原文 / TXT] --> G[分段与原文约束]
    G --> N
    N --> R[对照阅读与人工校对]
    R --> S[SQLite 保存 / 恢复 / 导出]
```

## 已测结果与边界

相同微调 Q8、相同输入与生成预算下，比较自由生成和输入约束生成。下表为开发选择后冻结的增量保留断句集，bench/ood 各 20 条；它来自现有来源语料，不是新建独立 benchmark，参考没有逐题专家校点认证。

|题组|自由生成严格 F1|原文约束严格 F1|约束完整正文一致率|
|---|---:|---:|---:|
|bench|0.6829|0.8291|100%|
|ood|0.8172|0.9094|100%|
|全部样本均值|**0.7501**|**0.8693**|**100%**|

严格断句 F1 在正文变化时记 0，否则比较标点边界；它不是标点类型或语义准确率。**上表是解码策略收益，不是微调前后的收益**。正文一致是约束契约，不代表语义断句正确、引号平衡或翻译正确。

开发集补充了两示例基座对照，说明更强提示会缩小与微调模型的差距。Q8/Q4 对照、原始输出、协议、双种子结果和真实服务验收见 [验证报告](reports/validation/REPORT.md)。尚未完成独立人类翻译盲评，不能宣称语料完全无误、无灾难性遗忘或 Q4 全任务质量损失低于某个比例。

本机 500 / 1500 / 3000 字各一次的完整处理耗时约 12.83 / 34.96 / 69.55 秒，包含分段、推理和保存，仅是有限工作负载观测，不是 P95 或生产 SLA。

## 目录

```text
api/                  FastAPI、鉴权、任务接口与阅读页面
classical/            数据准备、训练、评分、约束、推理及文档保存
configs/              数据审核与实验检查配置
deploy/               可选的 Ollama 模型注册模板
scripts/              启停、导出、评测及真实服务检查
tests/                程序检查（部分需要另备本地资产）
reports/validation/   选定的当前结果、协议和验收记录
reports/gguf-artifacts.json  已测 GGUF 的大小、格式与 SHA-256
```

仓库不分发基座、adapter、GGUF、原始语料、运行数据库、访问令牌、下载工具和缓存。克隆后需要另外准备这些资产，不能直接启动已有模型。

## 先选择 PowerShell 7

项目的 `.ps1` 脚本使用 **PowerShell 7**。Windows 自带的 `powershell.exe` 通常运行 Windows PowerShell 5.1；打开 Windows Terminal 并不代表已经切换到 PowerShell 7。可先检查当前窗口：

```powershell
$PSVersionTable.PSVersion
```

如果本机已有模型和 Python 环境，选择下面一种方式设置 `$pwsh` 后，直接跳到[启动研读工具](#启动研读工具)，不必重新准备环境或训练。

**方式一：已经安装 PowerShell 7，且能找到 `pwsh.exe`。**

```powershell
$pwsh = (Get-Command pwsh.exe -CommandType Application -ErrorAction Stop).Source
```

**方式二：当前只有 Windows PowerShell 5.1，但安装了带本地运行时的 Codex 桌面版。** 在当前窗口执行下面这段，自动读取已安装运行时的位置，不写死随更新变化的安装路径：

```powershell
$runtime = Get-AppxPackage -Name 'OpenAI.CodexPrimaryRuntime*' |
    Sort-Object Version -Descending |
    Select-Object -First 1
if (-not $runtime) {
    throw '未找到 Codex 本地运行时，请按下方说明单独安装 PowerShell 7。'
}
$pwsh = Join-Path $runtime.InstallLocation 'dependencies\native\powershell\pwsh.exe'
if (-not (Test-Path -LiteralPath $pwsh)) {
    throw '当前 Codex 运行时没有这个 PowerShell 入口，请单独安装 PowerShell 7。'
}
```

两种方式都用下面的命令确认目标解释器版本，输出应为 `7.x`。后续示例通过 `& $pwsh -NoProfile -File ...` 显式调用它，因此也可以从当前的 5.1 窗口启动项目：

```powershell
& $pwsh -NoProfile -Command '$PSVersionTable.PSVersion.ToString()'
```

如果两种方式都不可用，可按[微软官方安装说明](https://learn.microsoft.com/en-us/powershell/scripting/install/install-powershell-on-windows)安装独立 PowerShell 7，例如在提供 WinGet 的系统上执行以下命令；安装后重新打开终端，再使用方式一。本项目的运行不要求安装 Codex。

```powershell
winget install --id Microsoft.PowerShell --source winget
```

`$pwsh` 只在当前窗口中保留，换窗口后需重新选择。脚本采用 UTF-8 编码，5.1 可能误读其中的中文并报“意外的标记”等解析错误，见[微软编码说明](https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_character_encoding)。遇到这种情况先确认目标解释器，而不是重新下载模型。

## 首次环境与资产准备

已验证 Windows、PowerShell 7、Python 3.12 和 NVIDIA CUDA 单卡。依赖的实测固定版本在 [requirements.txt](requirements.txt)，CUDA 版 PyTorch 应匹配本机驱动。先创建独立环境，按自己的 CUDA 条件安装 PyTorch，再安装其余依赖；不要覆盖已有共享环境。

以下环境、安装与训练命令均在仓库根目录执行。当前电脑的路径如下，其他机器请换成实际克隆位置：

```powershell
Set-Location 'D:\Projects\Qwen3_Classical_Optimized'
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
# 仅在新环境中显式选择这个 Python
$env:CLASSICAL_PYTHON = (Resolve-Path '.\.venv\Scripts\python.exe').Path
```

解释器优先读取 `CLASSICAL_PYTHON`，其次读取仅本机保留的 `data/python.path`、项目 `.venv`，最后使用 PATH 中的 Python。该本机路径文件不会上传。

运行资产的位置与用途：

|位置|内容|何时需要|
|---|---|---|
|`assets/base_model/`|对应 Qwen3-4B 基座及 tokenizer，包含完整精度权重|训练、合并、HF 推理、原生模板核对|
|`assets/source_dataset/`|有来源元数据的 train/dev JSONL 与 ood 子目录|数据准备与领域评测|
|`assets/bench/`|对应版本的断句/翻译参考题|数据重合检查与评测|
|`data/prepared/`|准备程序生成的冻结数据、tokenized JSONL 与 manifest|训练与受控对照|
|`weights/clean_2k_seed24/`|本机训练所得 PEFT adapter 与 tokenizer|HF 推理及合并|
|`weights/gguf/clean-q8_0.gguf`|与产物清单一致的默认模型|原生阅读服务|
|`tools/llama-cuda/`|匹配的原生 CUDA 运行时|约束断句与 Q8/Q4 推理|

语料预处理格式见 `classical/prepare.py`，数据不提供一键下载流程。已有 GGUF 清单绑定实际测量产物，重新训练后应在新实验名下记录与导出，用 `scripts/record_quant_artifacts.py` 更新自己产物的哈希；启动时会核对哈希，不应直接删除检查。

默认 CUDA 工具可通过以下脚本安装到项目目录；脚本固定下载包并核对 SHA-256，不改系统 CUDA。复现 GGUF 导出还需要匹配的 `convert_hf_to_gguf.py`、`gguf-py`、`conversion` 和 `llama-quantize.exe`，路径见 `scripts/Export-GGUF.ps1`，仓库不内嵌上游工具源码。

```powershell
& $pwsh -NoProfile -File .\scripts\Install-NativeCuda.ps1
```

模型、语料和第三方工具沿用各自的许可和使用条件，本仓库不对这些上游资产重新授权。

## 启动研读工具

本机已有资产时从这里开始。先完成上方的 PowerShell 选择，在同一窗口进入项目目录；其他机器请换成自己的仓库路径：

```powershell
Set-Location 'D:\Projects\Qwen3_Classical_Optimized'
```

已有环境通过 `data/python.path` 配置时，启动前无需重设 `CLASSICAL_PYTHON`。只有需要更换 Python 环境时才设置该变量，它的优先级最高。确认训练和其他占用显存的服务结束后，启动研读工具：

```powershell
& $pwsh -NoProfile -File .\scripts\Start-Reader.ps1 -OpenBrowser
```

等终端出现“研读工具已就绪”，浏览器会自动打开；也可以手动访问 [本地研读页面](http://127.0.0.1:8010)。若页面要求访问令牌，在同一窗口读取并复制到页面连接框：

```powershell
Get-Content -LiteralPath .\data\access.token -Encoding UTF8
```

令牌仅在本机使用。需要确认服务状态时可执行：

```powershell
Invoke-RestMethod 'http://127.0.0.1:8010/health'
```

默认脚本管理独立原生服务 `127.0.0.1:11436`，不使用默认 Ollama 端口。遇到端口占用会拒绝启动，不停止未知进程。

### 使用结束后停止服务

关闭页面后，模型服务仍可能占用显存。使用结束后在同一窗口单独执行停止命令：

```powershell
& $pwsh -NoProfile -File .\scripts\Stop-Reader.ps1
```

如果上次启动被中断，且再次启动提示已有本项目的进程记录，先执行上述停止命令清理并停止本项目记录的进程，再重新启动。换终端窗口后先重新设置 `$pwsh`。

### 阅读与模型选项

每次最多处理 3000 字符。结果保存在 `data/reader.sqlite`，页面“最近文档”可以恢复；只有完整结束或人工确认的段才算完成。部分处理时导出保留尚未处理的原文。整篇默认预算 300 秒，可用 `-TaskTimeoutSeconds` 调整。

断句的原文按 NFC 规范化逐字符固定，包含已有标点、数字和空白；每个间隙至多增加两个白名单中文标点。输入含无法可靠处理的控制/私用字符会明确拒绝。翻译和通用聊天维持普通生成，内容需要自行核对。

可用 `-Backend hf` 显式切换到已准备的 HF NF4 模型（先停止原生服务）；`-Unconstrained` 仅供关闭阅读断句约束的诊断。两种后端不能同时占用当前 GPU。

## 训练与验证

训练与现有权重隔离，非空输出目录会拒绝覆盖。默认 rank=32，适配 q/k/v/o、gate/up/down 七类投影；NF4 + double quant、BF16 计算、batch=1、梯度累积=8、梯度检查点，只监督答案 token。数据先检查来源和参考，再训练；不能用形式检查代替语义审核。

```powershell
# 首先准备数据；不覆盖已冻结目录
& $pwsh -NoProfile -File .\scripts\Run-Python.ps1 -m classical.prepare
# 训练前只验证冻结文件与标签，不加载 GPU
& $pwsh -NoProfile -File .\scripts\Run-Python.ps1 -m classical.train --name reading_seed24 --validate-only
# 真正训练使用唯一实验名
& $pwsh -NoProfile -File .\scripts\Run-Python.ps1 -m classical.train --name reading_seed24 --seed 24
# 程序检查：部分数据/权重检查需准备对应本地资产
& $pwsh -NoProfile -File .\scripts\Run-Python.ps1 -m unittest discover -s tests -v
```

领域评分、同 checkpoint 量化比较和真实服务检查分别由 `classical.evaluate`、`classical.quant_evaluate`、`scripts/evaluate_native.py`、`scripts/validate_native_contract.py`、`scripts/validate_native_documents.py` 实现。现有报告的题目已使用，不能在下一次选方案时仍当作未见测试。

真实浏览器检查还需要 `requirements-test.txt` 和已安装的 Edge，由 `scripts/validate_browser.py` 在项目缓存目录建立隔离配置；不会读取日常浏览器配置。程序检查、服务验收和模型质量分开记录。

## 接口与运行限制

|接口|用途|
|---|---|
|`GET /health`|加载状态与后端信息|
|`POST /api/process`|分段断句/翻译与恢复|
|`POST /api/cancel`|按 request_id 取消任务|
|`GET /api/progress/{request_id}`|段数与实际用时|
|`GET /api/documents`|最近文档|
|`GET /api/documents/{key}`|恢复已保存文档|
|`PATCH /api/documents/{key}/segments/{index}`|携带 revision 保存校对|
|`POST /v1/chat/completions`|OpenAI 兼容非流式与 SSE 聊天|

业务接口使用本机 Bearer Token；这是单机访问控制，不是完整多用户账号系统。实际生成并发固定为 1，等待队列默认 4，满队列返回 429。取消/截止后需确认原生槽位空闲才释放容量，不能确认则隔离引擎并要求重启；不宣称 GPU 内核即时停止。使用单个 Uvicorn worker，不能通过增加 worker 直接复制模型扩容。
