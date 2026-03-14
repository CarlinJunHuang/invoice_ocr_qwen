# OCR Comparison Report Design

## Goal

Extend the existing `invoice_ocr_glm_paddle_eval` experiment so one compare run can evaluate multiple OCR/VLM frontends, preserve each model's raw output and invoice JSON, and generate a human-readable comparison report for side-by-side review.

## Scope

This design keeps the current POC boundaries:

- Input remains image-only (`png`, `jpg`, `jpeg`)
- Extraction scope remains invoice / credit note fields only
- Evidence grounding continues to rely on EasyOCR line boxes
- Local-only execution, with all model downloads and caches kept under the repo runtime directory

The new work adds:

- `FireRed-OCR` as another OCR frontend mode
- Run-level Markdown and HTML comparison reports
- A stitched bbox overlay montage per input image for quick visual comparison

## Recommended Approach

Use the existing experiment structure and treat `FireRed-OCR` as another frontend mode inside the same pipeline.

For each compare run:

1. Run each selected mode through the same `run_extract` flow
2. Save the existing per-mode artifacts unchanged
3. Build a run-level report from those saved artifacts
4. Build a montage image that lays out each mode's overlay side by side

This keeps the report logic downstream of extraction, so it can compare any future mode without changing the core output contract.

## Alternatives Considered

### 1. Report-first wrapper outside the pipeline

Pros:
- Smallest change to the current extraction code

Cons:
- Requires re-reading loosely structured files from disk
- Harder to evolve when artifact names or schema change

### 2. Integrate reporting into `run_compare`

Pros:
- Direct access to in-memory summaries
- Minimal duplicate parsing

Cons:
- Slightly couples reporting to compare execution

This is the recommended option because it stays simple and matches the current POC style.

### 3. Export only a single stitched image

Pros:
- Fast to scan visually

Cons:
- Not enough for JSON and raw OCR review

This is useful as a supporting artifact, not as the primary deliverable.

## Data Flow

The updated compare flow will be:

1. `run_compare` iterates through selected modes and images
2. Each mode still writes:
   - `raw_model_output.txt`
   - `invoice_fields.json`
   - `grounded_boxes.json`
   - `page_01_overlay.png`
   - `run_summary.json`
3. A report builder scans the compare results and produces:
   - `reports/<run_name>/report.md`
   - `reports/<run_name>/report.html`
   - `reports/<run_name>/<image_stem>_overlay_montage.png`

## FireRed-OCR Integration

`FireRed-OCR` should be loaded through the same Transformers-based runtime used by the current models, with its own mode kind and prompt preparation logic.

The implementation should:

- Keep model ID and generation settings in `configs/default.yaml`
- Reuse the existing local Hugging Face cache directory
- Preserve token and latency metrics in `run_summary.json`
- Fall back cleanly to an empty envelope if the mode fails

## Error Handling

The report must remain robust if any single mode fails.

- Failed modes should still appear in the report
- The report table should show the error message and missing artifact state
- Montage generation should skip missing overlays instead of failing the whole run

## Testing Strategy

Keep validation lightweight:

- Smoke-run one image across a subset of modes
- Verify `benchmark_summary.json`, `report.md`, `report.html`, and montage image are created
- Verify `FireRed-OCR` artifacts follow the same contract as other modes

## Success Criteria

The change is successful if one compare command can:

- run `GLM-OCR`, `PaddleOCR-VL`, `PaddleOCR-VL-1.5`, and `FireRed-OCR`
- optionally run parser-backed variants that keep invoice JSON output
- produce a report that makes raw OCR, JSON output, metrics, and bbox overlays easy to compare
