# DeepSeek OCR Raw Implementation Plan

**Goal:** Add a raw-only `DeepSeek-OCR` mode to the local OCR comparison experiment.

**Architecture:** Extend the current model runner with one DeepSeek-specific inference path based on the official `model.infer(...)` flow, store official native artifacts in the mode output directory, and surface them through the existing raw comparison report.

**Tech Stack:** Python, Transformers, Hugging Face remote code, existing reporting pipeline

---

### Task 1: Add the DeepSeek mode contract

**Files:**
- Modify: `D:\OneDrive\学在南科3.0\Internship\KAPPS Consulting\QwenOCRTest\invoice_ocr_qwen\invoice_ocr_glm_paddle_eval\configs\default.yaml`
- Modify: `D:\OneDrive\学在南科3.0\Internship\KAPPS Consulting\QwenOCRTest\invoice_ocr_qwen\invoice_ocr_glm_paddle_eval\src\invoice_ocr_glm_paddle_eval\models.py`

**Steps:**

1. Add `deepseek_ocr` to the mode config.
2. Add a DeepSeek-specific loader and inference path using the official prompt `"<image>\nFree OCR."`.
3. Ensure runtime metrics are captured in the same summary shape as the other OCR modes.

### Task 2: Preserve DeepSeek-native artifacts

**Files:**
- Modify: `D:\OneDrive\学在南科3.0\Internship\KAPPS Consulting\QwenOCRTest\invoice_ocr_qwen\invoice_ocr_glm_paddle_eval\src\invoice_ocr_glm_paddle_eval\pipeline.py`
- Modify: `D:\OneDrive\学在南科3.0\Internship\KAPPS Consulting\QwenOCRTest\invoice_ocr_qwen\invoice_ocr_glm_paddle_eval\src\invoice_ocr_glm_paddle_eval\models.py`

**Steps:**

1. Let the model runner write DeepSeek-native artifacts into a dedicated subdirectory inside the current mode output folder.
2. Detect primary raw output plus any native JSON or image artifacts.
3. Save those artifact paths into `run_summary.json`.

### Task 3: Surface DeepSeek-native artifact availability in reports

**Files:**
- Modify: `D:\OneDrive\学在南科3.0\Internship\KAPPS Consulting\QwenOCRTest\invoice_ocr_qwen\invoice_ocr_glm_paddle_eval\src\invoice_ocr_glm_paddle_eval\reporting.py`
- Modify: `D:\OneDrive\学在南科3.0\Internship\KAPPS Consulting\QwenOCRTest\invoice_ocr_qwen\invoice_ocr_glm_paddle_eval\README.md`

**Steps:**

1. Make sure raw reports show DeepSeek in the same layout as the current OCR modes.
2. Show whether native bbox or native overlay files were actually produced.
3. Keep the wording explicit that no parser-based overlay is used in the raw report.

### Task 4: Verify DeepSeek-OCR locally

**Files:**
- No source changes required unless verification finds defects

**Steps:**

1. Run `deepseek_ocr` on one existing internal test image.
2. Run a compare that includes `deepseek_ocr`.
3. Confirm:
   - `raw_model_output.txt` exists
   - `run_summary.json` includes DeepSeek metrics
   - raw report files are generated
   - native artifact availability is correctly reported
