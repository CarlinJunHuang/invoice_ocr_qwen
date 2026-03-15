# dots.ocr 1.5 Tryout Implementation Plan

**Goal:** Add an isolated `dots_ocr_tryout` folder that tests `dots.ocr 1.5` with HF first and then `vLLM + docker` if needed, preserving only raw output.

**Architecture:** Keep the tryout independent from the main OCR experiment. Use one tiny folder with setup and run scripts, capture raw output and logs, and do not introduce parser or overlay logic.

**Tech Stack:** PowerShell, Python, Hugging Face local inference, optional Docker and vLLM

---

### Task 1: Create the isolated tryout folder

**Files:**
- Create: `D:\OneDrive\学在南科3.0\Internship\KAPPS Consulting\QwenOCRTest\invoice_ocr_qwen\dots_ocr_tryout\README.md`
- Create: `D:\OneDrive\学在南科3.0\Internship\KAPPS Consulting\QwenOCRTest\invoice_ocr_qwen\dots_ocr_tryout\pyproject.toml`
- Create: `D:\OneDrive\学在南科3.0\Internship\KAPPS Consulting\QwenOCRTest\invoice_ocr_qwen\dots_ocr_tryout\.gitignore`
- Create: `D:\OneDrive\学在南科3.0\Internship\KAPPS Consulting\QwenOCRTest\invoice_ocr_qwen\dots_ocr_tryout\scripts\bootstrap_hf.ps1`
- Create: `D:\OneDrive\学在南科3.0\Internship\KAPPS Consulting\QwenOCRTest\invoice_ocr_qwen\dots_ocr_tryout\scripts\run_hf.ps1`
- Create: `D:\OneDrive\学在南科3.0\Internship\KAPPS Consulting\QwenOCRTest\invoice_ocr_qwen\dots_ocr_tryout\scripts\run_vllm_docker.ps1`
- Create: `D:\OneDrive\学在南科3.0\Internship\KAPPS Consulting\QwenOCRTest\invoice_ocr_qwen\dots_ocr_tryout\scripts\collect_result.ps1`

**Steps:**

1. Create the folder and ignore runtime/output artifacts.
2. Keep setup and run commands explicit and short.
3. Make sure the folder can be deleted independently.

### Task 2: Add HF raw inference path

**Files:**
- Create: `D:\OneDrive\学在南科3.0\Internship\KAPPS Consulting\QwenOCRTest\invoice_ocr_qwen\dots_ocr_tryout\run_hf.py`

**Steps:**

1. Load the chosen public model source.
2. Run one image through the simplest inference path available.
3. Save raw output and error logs under `outputs/hf/<run_name>/`.

### Task 3: Add vLLM plus docker fallback

**Files:**
- Create: `D:\OneDrive\学在南科3.0\Internship\KAPPS Consulting\QwenOCRTest\invoice_ocr_qwen\dots_ocr_tryout\run_vllm_client.py`

**Steps:**

1. Add a script that starts or references a local Docker-based vLLM server.
2. Send one image request and save the raw output.
3. Preserve server logs or command logs alongside the output.

### Task 4: Verify and summarize

**Files:**
- Modify: `D:\OneDrive\学在南科3.0\Internship\KAPPS Consulting\QwenOCRTest\invoice_ocr_qwen\dots_ocr_tryout\README.md`

**Steps:**

1. Attempt the HF path first.
2. If HF fails or gives unusable output, attempt the vLLM path.
3. Record the final status and exact output locations in the README.
