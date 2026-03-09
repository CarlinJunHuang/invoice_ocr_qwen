# Invoice OCR Qwen

[English README](README.md)

这个仓库现在包含两条并行的发票抽取实验路线：

- `ocr-first`：OCR -> 结构化抽取 -> evidence grounding -> overlay
- `direct-vlm`：单个视觉模型 -> 直接输出 fields + bbox JSON -> overlay

其中 `direct-vlm` 是最近 direct bench 的公开整理版。这里保留了可公开的代码、prompt 档位和输出合同，不包含任何私有样本产物、API key 或本地绝对路径。

## 两条路线

### 1. OCR-first 主线

这是仓库里原本就有的路线。

- Qwen-VL 可以消费做过 OCR 的页面图片
- Qwen 3.5 可以消费 OCR 文本
- evidence 会回挂到 OCR 行
- 最终产物包括 `envelope.json`、`grounded_evidence.json` 和 overlay 图

主要命令：

```powershell
.\scripts\run_extract.ps1 -Mode qwen3_vl_bf16 -InputImages .\input\invoices\page1.png
.\scripts\run_benchmark.ps1 -Modes qwen3_vl_bf16,qwen3_5_bf16 -InputImages .\input\invoices\page1.png
```

### 2. Direct VLM 路线

这是新增的单模型路线。

- 单张图片或 PDF 直接喂给一个多模态模型
- 要求模型一次性返回发票字段和 bounding boxes
- 本地再做归一化和画框
- 输出一份 envelope-like 结果和一份 bbox-rich 调试结果

这条路线故意不做 evidence mapping，目的就是先快速验证小模型或 API 模型的效果，再决定后续是否更深地并回主线。

Python 入口：

- `python -m invoice_ocr_qwen.direct_bench`
- `python -m invoice_ocr_qwen.direct_bench.dataset`

PowerShell 包装脚本：

- `.\scripts\run_direct_extract.ps1`
- `.\scripts\run_direct_dataset.ps1`

## Prompt 档位

Prompt 文件放在 `src/invoice_ocr_qwen/direct_bench/prompts/`。

- `core_fields.txt`
  - 极简 smoke prompt
- `poc_invoice_fields.txt`
  - 扩展字段版
- `poc_invoice_fields_v2.txt`
  - 改进后的字段版
- `poc_invoice_contract_v1.txt`
  - baseline direct 合同
- `poc_invoice_contract_v2.txt`
  - 更强调 null-over-guess 的版本
- `poc_invoice_contract_v3.txt`
  - 当前整理后的 refined prompt
  - 也是本仓库 direct bench 的默认 prompt

可以把 `v1` 理解成公开迁移的基础版本，把 `v3` 理解成当前更实用的改进版本。

## 安装

在仓库根目录执行：

```powershell
.\scripts\bootstrap.ps1
```

运行时默认放在仓库外部的 `..\runtime\`，避免模型缓存和虚拟环境污染仓库本身。

## Direct VLM 示例

本地 Ollama：

```powershell
.\scripts\run_direct_extract.ps1 `
  -Backend ollama `
  -Model qwen3-vl:4b-instruct-q8_0 `
  -InputFile .\input\invoices\invoice_a.png `
  -RunName direct-q3vl-4b
```

兼容 OpenAI 的 API：

```powershell
$env:QWEN_API_BASE_URL = "https://your-endpoint.example/v1"
$env:QWEN_API_KEY = "YOUR_API_KEY"

.\scripts\run_direct_extract.ps1 `
  -Backend openai-compatible `
  -Model qwen3.5-flash `
  -InputFile .\input\invoices\invoice_a.png `
  -RunName direct-q35f-api
```

批量对比：

```powershell
.\scripts\run_direct_dataset.ps1 `
  -Backend ollama `
  -RunPrefix direct-batch-v3 `
  -Models qwen3-vl:2b-instruct-q8_0,qwen3-vl:4b-instruct-q8_0 `
  -InputFiles .\input\invoices\invoice_a.png,.\input\invoices\invoice_b.png
```

## Direct VLM 关键参数

这条路线当前支持的实用参数：

- `--backend`：`ollama` 或 `openai-compatible`
- `--model`：模型 id
- `--prompt-file`：prompt 文件，默认是 `poc_invoice_contract_v3.txt`
- `--temperature`：默认 `0.0`
- `--max-tokens`：单跑默认 `2400`，批量 runner 默认 `3072`
- `--timeout-sec`
- `--dpi`
- `--max-pages`
- `--max-long-side`
- `--max-pixels`
- `--jpeg-quality`
- `--allow-thinking-fallback`：只给本地模型兜底用，如果模型把答案塞进 `thinking` 而不是 `content`

## 输出目录

OCR-first 输出：

- `output/<run_name>/<mode>/`

Direct VLM 输出：

- `output/direct/<run_name>/`

Direct 单跑产物：

- `prompt.txt`
- `page_raw_outputs.json`
- `page_model_metadata.json`
- `parsed_output.json`
- `envelope.json`
- `run_summary.json`
- `page_XX_overlay.png`
- `page_XX_line_items_overlay.png`

Direct 批量产物：

- `dataset-summary.tsv`
- `dataset-summary.jsonl`
- `dataset-aggregate.tsv`

## Direct 输出合同

Direct bench 的最终输出是 envelope-like 结构，包含：

- `request_id`
- `schema_version`
- `doc_type`
- `extracted`
- `clauses`
- `eligibility`
- `warnings`
- `errors`

和 OCR-first 主线的一个重要差异：

- direct bench 不生成 evidence 对象
- bbox 调试信息放在 `parsed_output.json`

## 为 direct bench 新增的依赖

这次额外加了几个运行时依赖：

- `numpy`
- `requests`
- `pypdfium2`

它们只服务于 direct 路线的页面渲染、API 调用和 bbox 处理。

## 隐私说明

当前提交只包含可公开复用的内容：

- 不包含私有样本输出
- 不包含 API key
- 文档里不保留本地绝对路径
