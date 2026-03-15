# Invoice OCR GLM Paddle Eval

这个目录用于做一个尽量简单、可本地运行的 OCR / 文档抽取实验，目标是快速比较几种本地模型在 invoice 图片上的效果。

当前范围：
- 输入：`png / jpg / jpeg`
- 文档类型：`Invoice / Credit Note`
- 输出：原始 OCR 输出、结构化 JSON、evidence grounding、可视化 overlay、对比报告
- 运行方式：完全本地，不依赖云 API

当前不覆盖：
- PDF
- PO / Transport Document
- Clause Detection
- Financing Eligibility

## 模式

基础 OCR / VLM 模式：
- `glm_ocr`
- `deepseek_ocr`
- `paddleocr_vl_v1`
- `paddleocr_vl_1_5`
- `firered_ocr`

接 `Qwen3.5` parser 的组合模式：
- `glm_ocr_qwen3_5_2b`
- `glm_ocr_qwen3_5_4b`
- `paddleocr_vl_v1_qwen3_5_2b`
- `paddleocr_vl_v1_qwen3_5_4b`
- `paddleocr_vl_1_5_qwen3_5_2b`
- `paddleocr_vl_1_5_qwen3_5_4b`
- `firered_ocr_qwen3_5_2b`
- `firered_ocr_qwen3_5_4b`

说明：
- 基础模式：模型直接从图片输出 OCR / Markdown / layout 结果
- 组合模式：先做 OCR，再把原始结果交给 `Qwen3.5` 生成目标 JSON

## 目录

- `private_inputs/invoices/`
  - 内部测试图片目录，不上传到 Git
- `outputs/`
  - 单次抽取和 compare 的原始产物目录，不上传到 Git
- `reports/`
  - compare 自动生成的 Markdown / HTML 报告，不上传到 Git
- `.local_runtime/`
  - 本实验自己的虚拟环境、模型缓存、torch 缓存、临时目录
- `configs/default.yaml`
  - 模式配置
- `scripts/`
  - 安装和运行脚本
- `src/`
  - Python 实现

## 本地缓存

大文件都放在当前目录下面：
- 虚拟环境：`.local_runtime/.venv311`
- Hugging Face 缓存：`.local_runtime/hf-cache`
- Torch 缓存：`.local_runtime/torch-cache`
- 临时目录：`.local_runtime/tmp`

## 安装

```powershell
.\scripts\bootstrap.ps1
```

## 运行

单图、单模式：

```powershell
.\scripts\run_extract.ps1 `
  -Mode glm_ocr `
  -InputImages .\private_inputs\invoices\1a.png
```

多模式对比：

```powershell
.\scripts\run_compare.ps1 `
  -Modes glm_ocr,paddleocr_vl_v1,paddleocr_vl_1_5,firered_ocr `
  -InputImages .\private_inputs\invoices\1a.png
```

## 输出说明

单次抽取目录：

```text
outputs/<run_name>/<image_name>/<mode>/
|- raw_model_output.txt
|- invoice_fields.json
|- grounded_boxes.json
|- page_01_overlay.png
\- run_summary.json
```

compare 报告目录：

```text
reports/<run_name>/
|- report.md
|- report.html
\- *_overlay_montage.png
```

## 当前观察

- `GLM / Paddle / FireRed` 已经能稳定纳入同一条本地 compare 流程
- `DeepSeek-OCR` 的 HF remote-code 兼容问题较多，当前更适合单独隔离验证
- 这条实验链路的重点是可读的原始输出、可回溯的 JSON 和可解释的 overlay，而不是一次性追求最复杂的抽取逻辑
