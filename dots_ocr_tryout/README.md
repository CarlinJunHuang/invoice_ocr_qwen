# dots.ocr 1.5 Tryout

这个目录只做一件事：验证 `dots.ocr 1.5` 能否在本机稳定输出原始 layout JSON 和原生 bbox。

当前策略：
- 先试 Hugging Face 本地推理
- 如果 HF 路径失败，再试 `vLLM + Docker`
- 只保留原始输出，不接 parser，不做字段抽取，不做额外后处理

## 目录

- `.local_runtime/`
  - 本 tryout 自己的虚拟环境、缓存和权重目录
- `outputs/`
  - 每次运行的原始输出和状态文件，不上传到 Git
- `scripts/`
  - 启动脚本
- `run_hf.py`
  - HF 本地推理脚本
- `run_vllm_client.py`
  - 调用本地 vLLM OpenAI 接口的最小客户端

## 运行

初始化环境：

```powershell
.\scripts\bootstrap_hf.ps1
```

HF 本地推理：

```powershell
.\scripts\run_hf.ps1 -InputImage ..\invoice_ocr_glm_paddle_eval\private_inputs\invoices\1a.png
```

vLLM + Docker：

```powershell
.\scripts\run_vllm_docker.ps1 -InputImage ..\invoice_ocr_glm_paddle_eval\private_inputs\invoices\1a.png
```

默认的 vLLM 参数针对 `RTX 4070 12GB` 做了保守收敛：
- `max_model_len = 4096`
- `gpu_memory_utilization = 0.90`
- `max_tokens = 3000`

第一次启动 vLLM 需要几分钟做模型加载和编译，后续请求会快很多。

## 输出

HF 输出目录：

```text
outputs/hf/<run_name>/
|- raw_output.txt
|- status.json
|- metadata.json
\- error.txt
```

vLLM 输出目录：

```text
outputs/vllm/<run_name>/
|- request.json
|- response.json
|- raw_output.txt
\- status.json
```

## 当前结论

- HF 路径在这台 Windows 本地环境里没有跑通，卡在 `flash_attn` 依赖
- `vLLM + Docker` 路径可以跑通，但需要把上下文长度压到更适合 12GB 显存的范围
- 在当前两张内部 invoice 样例图上，`dots.ocr 1.5` 的原始 layout 输出和原生 bbox 明显优于这轮对比过的其他 OCR 模型

权重目录使用 `DotsOCR_1_5` 这种不带句点的名字，避免官方 README 提到的路径兼容问题。
