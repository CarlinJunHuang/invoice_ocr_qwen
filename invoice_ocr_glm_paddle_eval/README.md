# Invoice OCR GLM Paddle Eval

这个目录用于做一个尽量简单的本地 POC，目标是验证几条本地可部署文档抽取路径在发票图片上的可用性。

当前范围：
- 输入：`png / jpg / jpeg`
- 文档类型：`Invoice / Credit Note`
- 输出：结构化 JSON、evidence grounding、bbox / overlay
- 本地运行，不依赖云 API

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

接 `Qwen3.5` 解析的组合模式：
- `glm_ocr_qwen3_5_2b`
- `glm_ocr_qwen3_5_4b`
- `paddleocr_vl_v1_qwen3_5_2b`
- `paddleocr_vl_v1_qwen3_5_4b`
- `paddleocr_vl_1_5_qwen3_5_2b`
- `paddleocr_vl_1_5_qwen3_5_4b`

说明：
- 基础模式：模型直接从图片输出 OCR 文本，脚本再做一个轻量解析。
- 组合模式：先让 `GLM-OCR / PaddleOCR-VL` 从图片读出文本，再把 OCR 文本交给 `Qwen3.5` 生成目标 JSON。

## 目录说明

- `private_inputs/invoices/`
  - 内部测试图片目录，不会上传到 Git
- `outputs/`
  - 运行产物目录，不会上传到 Git
- `.local_runtime/`
  - 本实验自己的虚拟环境、模型缓存、remote-code 缓存、临时目录
- `configs/default.yaml`
  - 模式配置
- `scripts/`
  - 安装和运行脚本
- `src/`
  - Python 实现

## 缓存位置

大文件都放在当前目录下：
- 虚拟环境：`.local_runtime/.venv311`
- Hugging Face 缓存：`.local_runtime/hf-cache`
- Torch 缓存：`.local_runtime/torch-cache`
- 临时目录：`.local_runtime/tmp`

如果之后不需要这个实验，直接删除整个 `invoice_ocr_glm_paddle_eval/` 即可。

## 安装

在当前目录执行：

```powershell
.\scripts\bootstrap.ps1
```

## 运行

单张图，单模式：

```powershell
.\scripts\run_extract.ps1 `
  -Mode glm_ocr `
  -InputImages .\private_inputs\invoices\1a.png
```

单张图，`GLM-OCR + Qwen3.5-2B`：

```powershell
.\scripts\run_extract.ps1 `
  -Mode glm_ocr_qwen3_5_2b `
  -InputImages .\private_inputs\invoices\1a.png
```

单张图，`PaddleOCR-VL 1.5 + Qwen3.5-2B`：

```powershell
.\scripts\run_extract.ps1 `
  -Mode paddleocr_vl_1_5_qwen3_5_2b `
  -InputImages .\private_inputs\invoices\1a.png
```

同一张图，对比多个模式：

```powershell
.\scripts\run_compare.ps1 `
  -Modes glm_ocr,paddleocr_vl_v1,paddleocr_vl_1_5,glm_ocr_qwen3_5_2b `
  -InputImages .\private_inputs\invoices\1a.png
```

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
  - OCR / VLM 模型直接输出的原始文本
- `parser_output.txt`
  - Qwen3.5 parser 的原始输出
- `run_summary.json`
  - 包含 OCR、主模型、Qwen parser 的耗时和 input/output token 统计
- `invoice_fields.json`
  - 最终结构化结果
- `grounded_boxes.json`
  - 根据 evidence 回挂到 OCR 行框后的 bbox
- `page_01_overlay.png`
  - 高亮结果图

## 当前实现方式

为保持这个 POC 简单，bbox 不依赖模型原生坐标输出：
1. 先用本地 EasyOCR 跑出 OCR 行文本和 bbox
2. 再让 `GLM-OCR / PaddleOCR-VL` 从图片读 OCR 文本
3. 可选地把 OCR 文本交给 `Qwen3.5` 解析成目标 JSON
4. 最后根据 evidence 文本回挂到 OCR 行框，生成 overlay

## 参考

- [GLM-OCR](https://huggingface.co/zai-org/GLM-OCR)
- [PaddleOCR-VL](https://huggingface.co/PaddlePaddle/PaddleOCR-VL)
- [PaddleOCR-VL-1.5](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5)
- [PaddleOCR-VL-hf](https://huggingface.co/merve/PaddleOCR-VL-hf)
- [PaddleOCR-VL-1.5-hf](https://huggingface.co/merve/PaddleOCR-VL-1.5-hf)
- [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)
