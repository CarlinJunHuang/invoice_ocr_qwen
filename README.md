# Invoice OCR Qwen

[中文说明](README.zh-CN.md)

This repository now contains two parallel experiment tracks for invoice extraction:

- `ocr-first`: OCR -> structured extraction -> evidence grounding -> overlay
- `direct-vlm`: single vision model -> fields + bbox JSON -> overlay

The `direct-vlm` track is a sanitized public port of the recent direct bench work. It keeps the public code, prompt profiles, and output contract, but does not include any private sample artifacts or local-only paths.

## Tracks

### 1. OCR-first pipeline

This is the original pipeline already present in the repo.

- Qwen-VL can consume page images after OCR
- Qwen 3.5 can consume OCR text
- evidence is grounded back to OCR lines
- final artifacts include `envelope.json`, `grounded_evidence.json`, and overlay images

Main commands:

```powershell
.\scripts\run_extract.ps1 -Mode qwen3_vl_bf16 -InputImages .\input\invoices\page1.png
.\scripts\run_benchmark.ps1 -Modes qwen3_vl_bf16,qwen3_5_bf16 -InputImages .\input\invoices\page1.png
```

### 2. Direct VLM bench

This is the newer single-model path.

- send one image or PDF directly to one multimodal model
- ask the model to return invoice fields and bounding boxes in one JSON payload
- normalize the result locally
- render overlays from returned boxes
- write one envelope-like output plus one bbox-rich debug output

This path intentionally does not do evidence mapping. It is meant for quick evaluation of smaller local or API models before deeper integration.

Main Python entry points:

- `python -m invoice_ocr_qwen.direct_bench`
- `python -m invoice_ocr_qwen.direct_bench.dataset`

PowerShell wrappers:

- `.\scripts\run_direct_extract.ps1`
- `.\scripts\run_direct_dataset.ps1`

## Prompt profiles

Prompt files live under `src/invoice_ocr_qwen/direct_bench/prompts/`.

- `core_fields.txt`
  - minimal smoke prompt
- `poc_invoice_fields.txt`
  - broader field-first prompt
- `poc_invoice_fields_v2.txt`
  - refined field prompt
- `poc_invoice_contract_v1.txt`
  - baseline direct contract
- `poc_invoice_contract_v2.txt`
  - stricter null-over-guess version
- `poc_invoice_contract_v3.txt`
  - current refined prompt
  - default for the direct bench in this repository

You can think of `v1` as the baseline public port and `v3` as the improved prompt profile.

## Install

From the repository root:

```powershell
.\scripts\bootstrap.ps1
```

The runtime is kept outside this repository under `..\runtime\` so model caches and virtualenvs do not pollute the repo.

## Direct VLM examples

Local Ollama:

```powershell
.\scripts\run_direct_extract.ps1 `
  -Backend ollama `
  -Model qwen3-vl:4b-instruct-q8_0 `
  -InputFile .\input\invoices\invoice_a.png `
  -RunName direct-q3vl-4b
```

OpenAI-compatible API:

```powershell
$env:QWEN_API_BASE_URL = "https://your-endpoint.example/v1"
$env:QWEN_API_KEY = "YOUR_API_KEY"

.\scripts\run_direct_extract.ps1 `
  -Backend openai-compatible `
  -Model qwen3.5-flash `
  -InputFile .\input\invoices\invoice_a.png `
  -RunName direct-q35f-api
```

Batch comparison:

```powershell
.\scripts\run_direct_dataset.ps1 `
  -Backend ollama `
  -RunPrefix direct-batch-v3 `
  -Models qwen3-vl:2b-instruct-q8_0,qwen3-vl:4b-instruct-q8_0 `
  -InputFiles .\input\invoices\invoice_a.png,.\input\invoices\invoice_b.png
```

## Direct VLM parameters

The direct bench supports these practical parameters:

- `--backend`: `ollama` or `openai-compatible`
- `--model`: model id
- `--prompt-file`: prompt profile file, defaulting to `poc_invoice_contract_v3.txt`
- `--temperature`: default `0.0`
- `--max-tokens`: default `2400` for one-off runs, `3072` in the dataset runner
- `--timeout-sec`
- `--dpi`
- `--max-pages`
- `--max-long-side`
- `--max-pixels`
- `--jpeg-quality`
- `--allow-thinking-fallback`: only for local models that leave `content` empty and put the answer in `thinking`

## Outputs

OCR-first outputs are written under:

- `output/<run_name>/<mode>/`

Direct VLM outputs are written under:

- `output/direct/<run_name>/`

Direct run artifacts:

- `prompt.txt`
- `page_raw_outputs.json`
- `page_model_metadata.json`
- `parsed_output.json`
- `envelope.json`
- `run_summary.json`
- `page_XX_overlay.png`
- `page_XX_line_items_overlay.png`

Direct batch artifacts:

- `dataset-summary.tsv`
- `dataset-summary.jsonl`
- `dataset-aggregate.tsv`

## Output contract for the direct bench

The direct bench returns an envelope-like structure with:

- `request_id`
- `schema_version`
- `doc_type`
- `extracted`
- `clauses`
- `eligibility`
- `warnings`
- `errors`

Important difference from the OCR-first pipeline:

- `evidence` is intentionally left empty in the direct bench
- bbox-rich debugging data is stored in `parsed_output.json`

## Dependencies added for the direct bench

The direct bench adds a few runtime dependencies:

- `numpy`
- `requests`
- `pypdfium2`

They are used only for page rendering, API calls, and bbox handling in the direct path.

## Privacy note

This repository only contains portable code, prompts, scripts, and docs.

- no private sample outputs
- no API keys
- no local absolute paths in the committed direct bench docs
