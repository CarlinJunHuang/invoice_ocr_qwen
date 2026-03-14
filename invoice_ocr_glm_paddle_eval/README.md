# Invoice OCR GLM Paddle Eval

这个目录用于做一个尽量简单、可本地运行的 OCR / 文档抽取 POC，目标是快速比较几种本地模型在发票图片上的效果。

当前范围：

- 输入：`png / jpg / jpeg`
- 文档范围：`Invoice / Credit Note`
- 输出：结构化 JSON、evidence grounding、bbox / overlay、原始输出、对比报告
- 运行方式：完全本地，不依赖云 API

当前不做：

- PDF
- PO / Transport Document
- Clause Detection
- Financing Eligibility

## 当前模式

基础 OCR / VLM 模式：

- `glm_ocr`
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

- 基础模式：模型直接从图片输出 OCR / Markdown 文本，再用轻量 parser 转成 invoice JSON
- 组合模式：先让 OCR / 文档模型输出文本，再把文本交给 `Qwen3.5` 生成目标 JSON

## 目录说明

- `private_inputs/invoices/`
  - 内部测试图片目录，不上传到 Git
- `outputs/`
  - 单次抽取和 compare 的原始产物目录，不上传到 Git
- `reports/`
  - compare 自动生成的 Markdown / HTML 对比报告，不上传到 Git
- `.local_runtime/`
  - 本实验自己的虚拟环境、模型缓存、torch 缓存、临时目录
- `configs/default.yaml`
  - 模式配置
- `scripts/`
  - 安装和运行脚本
- `src/`
  - Python 实现

## 缓存位置

大文件都放在当前目录下面：

- 虚拟环境：`.local_runtime/.venv311`
- Hugging Face 缓存：`.local_runtime/hf-cache`
- Torch 缓存：`.local_runtime/torch-cache`
- 临时目录：`.local_runtime/tmp`

如果后续不需要这个实验，直接删除整个 `invoice_ocr_glm_paddle_eval/` 即可。

## 安装

在当前目录执行：

```powershell
.\scripts\bootstrap.ps1
```

## 运行

单张图，单模式：

```powershell
.\scripts\run_extract.ps1 `
  -Mode firered_ocr `
  -InputImages .\private_inputs\invoices\1a.png
```

单张图，`FireRed-OCR + Qwen3.5-2B`：

```powershell
.\scripts\run_extract.ps1 `
  -Mode firered_ocr_qwen3_5_2b `
  -InputImages .\private_inputs\invoices\1a.png
```

同一张图，对比多个模式：

```powershell
.\scripts\run_compare.ps1 `
  -Modes glm_ocr,paddleocr_vl_v1,paddleocr_vl_1_5,firered_ocr,firered_ocr_qwen3_5_2b `
  -InputImages .\private_inputs\invoices\1a.png
```

如果直接使用默认模式，`run_compare.ps1` 会跑：

- `glm_ocr`
- `paddleocr_vl_v1`
- `paddleocr_vl_1_5`
- `firered_ocr`

## 输出内容

单次运行目录结构：

```text
outputs/<run_name>/<image_stem>/<mode>/
├── ocr_pages.json
├── raw_model_output.txt
├── parser_output.txt            # 仅组合模式存在
├── parsed_model_output.json
├── invoice_fields.json
├── grounded_boxes.json
├── page_01_overlay.png
└── run_summary.json
```

几个重点文件：

- `raw_model_output.txt`
  - OCR / 文档模型直接输出的原始文本
- `parser_output.txt`
  - Qwen3.5 parser 的原始输出
- `invoice_fields.json`
  - 最终结构化结果
- `grounded_boxes.json`
  - 根据 evidence 回挂到 EasyOCR 行框后的 bbox
- `page_01_overlay.png`
  - 当前模式的高亮图
- `run_summary.json`
  - OCR、主模型、parser 的耗时和 token 统计

## Compare 报告

执行 `run_compare.ps1` 后，还会生成：

```text
reports/<run_name>/
├── report.md
├── report.html
├── report_summary.json
└── <image_stem>_overlay_montage.png
```

报告内容包括：

- 每个模型的原始 OCR / Markdown 输出
- 最终 JSON
- 总耗时
- 主模型 token
- parser token
- grounded bbox 数量
- 每个模型自己的 overlay 图
- 同一张图的 overlay 拼图，方便横向比对

## 当前 bbox 方案

为了保持这个 POC 简单，bbox 不依赖模型原生坐标输出：

1. 先用本地 EasyOCR 跑出 OCR 行文本和 bbox
2. 再让 `GLM-OCR / PaddleOCR-VL / FireRed-OCR` 从图片读文本
3. 可选地把文本交给 `Qwen3.5` 生成目标 JSON
4. 最后根据 evidence 文本回挂到 EasyOCR 行框，生成 overlay

这样做的好处是每条链路都能落 explainable artifact，方便后续再升级成更精细的 bbox 方案。

## 参考

- [GLM-OCR](https://huggingface.co/zai-org/GLM-OCR)
- [PaddleOCR-VL](https://huggingface.co/PaddlePaddle/PaddleOCR-VL)
- [PaddleOCR-VL-1.5](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5)
- [PaddleOCR-VL-hf](https://huggingface.co/merve/PaddleOCR-VL-hf)
- [PaddleOCR-VL-1.5-hf](https://huggingface.co/merve/PaddleOCR-VL-1.5-hf)
- [FireRed-OCR](https://huggingface.co/FireRedTeam/FireRed-OCR)
- [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B)
- [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)
