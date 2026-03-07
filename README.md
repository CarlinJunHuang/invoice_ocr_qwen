# Invoice OCR Qwen

本仓库用于本地跑发票/票据类图片抽取实验，重点是把文档理解、结构化抽取、evidence 保留和 overlay 产物串成一条可复用的实验链路。

当前实现只依赖本地推理，不调用云 API，适合先在个人机器上比较不同模式，再决定后续如何并回主项目。

## 当前保留的模式

- `qwen3_vl_bf16`
- `qwen3_vl_4bit`
- `qwen3_5_bf16`
- `qwen3_5_4bit`
- `ocr_rules`，可选保底基线

说明：

- `Qwen3-VL` 走图片 + OCR 的文档理解路径
- `Qwen3.5` 走 OCR 文本抽取路径，便于和视觉路线做对比
- `4bit` 主要用于压显存，不保证比 `bf16` 更快
- `ocr_rules` 不是主方向，只用于快速检查 OCR、evidence grounding、overlay 这条链是否正常

## 输入和输出

输入图片目录：

- `.\input\invoices\`

支持格式：

- `.png`
- `.jpg`
- `.jpeg`

单次运行输出目录：

- `.\output\<run_name>\<mode>\`

重点产物：

- `envelope.json`
- `parsed_model_output.json`
- `raw_model_output.txt`
- `grounded_evidence.json`
- `page_01_overlay.png`
- `run_summary.json`

如果是 benchmark，还会额外生成：

- `benchmark_summary.json`

## 目录结构

```text
invoice_ocr_qwen/
├─ configs/
├─ input/
│  └─ invoices/
├─ output/
├─ scripts/
└─ src/
```

## 环境说明

本仓库默认把虚拟环境、模型缓存和 Torch 缓存写到仓库同级目录的 `..\runtime\`，目的是：

- 不把大模型缓存写进仓库
- 不把缓存写到系统默认目录
- 让同一工作区下的其他实验目录也能复用同一个运行时

默认位置：

- 虚拟环境：`..\runtime\.venv311`
- Hugging Face 缓存：`..\runtime\hf-cache`
- Torch 缓存：`..\runtime\torch-cache`

## 安装

在仓库根目录执行：

```powershell
.\scripts\bootstrap.ps1
```

这个脚本会：

- 创建 `..\runtime\.venv311`
- 安装 CUDA 版 PyTorch
- 安装项目依赖
- 尝试安装 `bitsandbytes`

## 如何准备图片

把待测图片放到：

- `.\input\invoices\`

建议命名方式：

- `invoice_a_page1.png`
- `invoice_a_page2.png`
- `invoice_b_page1.jpg`

多页文档直接按页顺序传入即可。

## 如何运行

### 1. 单跑 Qwen3-VL BF16

```powershell
.\scripts\run_extract.ps1 `
  -Mode qwen3_vl_bf16 `
  -InputImages .\input\invoices\page1.png
```

### 2. 单跑 Qwen3-VL 4bit

```powershell
.\scripts\run_extract.ps1 `
  -Mode qwen3_vl_4bit `
  -InputImages .\input\invoices\page1.png
```

### 3. 单跑 Qwen3.5 BF16

```powershell
.\scripts\run_extract.ps1 `
  -Mode qwen3_5_bf16 `
  -InputImages .\input\invoices\page1.png
```

### 4. 单跑 Qwen3.5 4bit

```powershell
.\scripts\run_extract.ps1 `
  -Mode qwen3_5_4bit `
  -InputImages .\input\invoices\page1.png
```

### 5. 同一批图片做模式对比

```powershell
.\scripts\run_benchmark.ps1 `
  -Modes qwen3_vl_bf16,qwen3_vl_4bit,qwen3_5_bf16,qwen3_5_4bit `
  -InputImages .\input\invoices\page1.png
```

### 6. 多页图片

```powershell
.\scripts\run_extract.ps1 `
  -Mode qwen3_vl_bf16 `
  -InputImages .\input\invoices\page1.png,.\input\invoices\page2.png
```

### 7. 直接走 Python 入口

```powershell
..\runtime\.venv311\Scripts\python.exe -m invoice_ocr_qwen.cli extract `
  --config .\configs\default.yaml `
  --mode qwen3_vl_bf16 `
  --input .\input\invoices\page1.png
```

## 推荐顺序

如果是 RTX 4070 12GB / 32GB RAM，建议先这样跑：

1. `qwen3_vl_bf16`
2. `qwen3_5_bf16`
3. 再补 `qwen3_vl_4bit`
4. 最后补 `qwen3_5_4bit`

如果只是先看整条链有没有跑通，可以先跑：

```powershell
.\scripts\run_extract.ps1 `
  -Mode ocr_rules `
  -InputImages .\input\invoices\page1.png
```

## 为什么会慢

本地文档抽取慢，主要不是因为“thinking”，而是下面几件事：

- 首次运行需要下载模型
- `Qwen3-VL` 处理整页图片时，视觉 token 开销本来就高
- `4bit` 主要省显存，不一定省时间
- `Qwen3.5` 路线虽然不看图，但仍然要先做 OCR，再跑结构化抽取

当前默认已经做了两件比较保守的加速处理：

- `Qwen3-VL` 视觉输入做了温和的像素上限控制，避免整页图片分辨率过高
- `Qwen3.5` 默认走确定性输出，尽量减少跑偏成大段自然语言的情况

如果发现小字丢失，可以把 `configs/default.yaml` 里的 `qwen3_vl_*` `max_pixels` 调高。

## 当前实现思路

整体流程保持和主项目方向一致：

1. 先做 OCR
2. 再做结构化抽取
3. 再把 evidence 回挂到 OCR 行框
4. 最后输出 envelope 和 overlay

这意味着：

- 即便某条模型路径拿不到原生 bbox，也仍然可以先保留 explainable evidence
- 后续要并回稳定 envelope / highlight 流程时，迁移成本会更低

## 参考

- [Qwen3-VL OCR Cookbook](https://github.com/QwenLM/Qwen3-VL/blob/main/cookbooks/ocr.ipynb)
- [Qwen3-VL Document Parsing Cookbook](https://github.com/QwenLM/Qwen3-VL/blob/main/cookbooks/document_parsing.ipynb)
