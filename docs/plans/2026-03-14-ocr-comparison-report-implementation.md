# OCR Comparison Report Implementation Plan

**Goal:** Add FireRed-OCR and run-level comparison reporting to the local OCR evaluation experiment.

**Architecture:** Extend the existing compare pipeline with one new OCR frontend mode and one downstream reporting stage. Preserve the current per-mode artifact layout so the report generator can summarize any mode consistently.

**Tech Stack:** Python, Transformers, Pillow, Markdown/HTML generation, existing PowerShell runners

---

### Task 1: Add the FireRed-OCR mode contract

**Files:**
- Modify: `D:\OneDrive\学在南科3.0\Internship\KAPPS Consulting\QwenOCRTest\invoice_ocr_qwen\invoice_ocr_glm_paddle_eval\configs\default.yaml`
- Modify: `D:\OneDrive\学在南科3.0\Internship\KAPPS Consulting\QwenOCRTest\invoice_ocr_qwen\invoice_ocr_glm_paddle_eval\src\invoice_ocr_glm_paddle_eval\models.py`

**Step 1: Add FireRed mode definitions**

- Add `firered_ocr` and optional parser-backed variants to the YAML config.
- Keep generation defaults conservative and local-GPU friendly.

**Step 2: Implement model loading and prompt preparation**

- Add a `firered_ocr` kind to the model loader.
- Reuse the local Hugging Face cache and remote code support if required.
- Return raw output plus token/latency metrics using the same summary shape.

**Step 3: Smoke-check a single-mode load path**

Run:

```powershell
python -m invoice_ocr_glm_paddle_eval.cli extract --config .\invoice_ocr_glm_paddle_eval\configs\default.yaml --mode firered_ocr --input .\invoice_ocr_glm_paddle_eval\private_inputs\invoices\1a.png
```

Expected:
- Artifacts are written
- `run_summary.json` includes `main_model`

**Step 4: Commit**

```bash
git add invoice_ocr_glm_paddle_eval/configs/default.yaml invoice_ocr_glm_paddle_eval/src/invoice_ocr_glm_paddle_eval/models.py
git commit -m "feat: add FireRed OCR mode"
```

### Task 2: Add report and montage generation

**Files:**
- Create: `D:\OneDrive\学在南科3.0\Internship\KAPPS Consulting\QwenOCRTest\invoice_ocr_qwen\invoice_ocr_glm_paddle_eval\src\invoice_ocr_glm_paddle_eval\reporting.py`
- Modify: `D:\OneDrive\学在南科3.0\Internship\KAPPS Consulting\QwenOCRTest\invoice_ocr_qwen\invoice_ocr_glm_paddle_eval\src\invoice_ocr_glm_paddle_eval\pipeline.py`

**Step 1: Build a run-level summary model**

- Read compare results and normalize the key fields needed by the report.
- Include image path, mode, error state, metrics, artifact paths, and selected extracted fields.

**Step 2: Render Markdown and HTML reports**

- Write `report.md` with a top summary table and per-mode sections.
- Write `report.html` with the same content for easier browsing.

**Step 3: Render montage images**

- Stitch per-mode overlay images horizontally for each input image.
- Skip missing overlays gracefully.

**Step 4: Wire reporting into `run_compare`**

- Generate reports after all mode runs complete.
- Save report artifact paths into `benchmark_summary.json`.

**Step 5: Commit**

```bash
git add invoice_ocr_glm_paddle_eval/src/invoice_ocr_glm_paddle_eval/reporting.py invoice_ocr_glm_paddle_eval/src/invoice_ocr_glm_paddle_eval/pipeline.py
git commit -m "feat: add OCR comparison reports"
```

### Task 3: Update CLI and docs

**Files:**
- Modify: `D:\OneDrive\学在南科3.0\Internship\KAPPS Consulting\QwenOCRTest\invoice_ocr_qwen\invoice_ocr_glm_paddle_eval\README.md`
- Modify: `D:\OneDrive\学在南科3.0\Internship\KAPPS Consulting\QwenOCRTest\invoice_ocr_qwen\invoice_ocr_glm_paddle_eval\scripts\run_compare.ps1`

**Step 1: Document FireRed and the new report outputs**

- Add FireRed mode names
- Add report artifact locations
- Keep wording straightforward and repo-neutral

**Step 2: Keep compare runner aligned**

- Ensure the compare script examples mention report output and the new mode where appropriate

**Step 3: Commit**

```bash
git add invoice_ocr_glm_paddle_eval/README.md invoice_ocr_glm_paddle_eval/scripts/run_compare.ps1
git commit -m "docs: document OCR comparison reporting"
```

### Task 4: Verify end-to-end behavior

**Files:**
- No source changes required unless verification finds a defect

**Step 1: Run a focused compare**

Run:

```powershell
python -m invoice_ocr_glm_paddle_eval.cli compare --config .\invoice_ocr_glm_paddle_eval\configs\default.yaml --modes glm_ocr paddleocr_vl_v1 firered_ocr --input .\invoice_ocr_glm_paddle_eval\private_inputs\invoices\1a.png --run-name compare-smoke
```

Expected:
- `benchmark_summary.json` exists
- `reports/compare-smoke/report.md` exists
- `reports/compare-smoke/report.html` exists
- `reports/compare-smoke/1a_overlay_montage.png` exists when overlays are available

**Step 2: Fix any defects from the smoke run**

- Keep fixes minimal and local to the failing stage

**Step 3: Commit**

```bash
git add .
git commit -m "fix: stabilize OCR comparison smoke run"
```
