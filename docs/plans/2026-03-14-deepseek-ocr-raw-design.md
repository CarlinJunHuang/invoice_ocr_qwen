# DeepSeek OCR Raw Integration Design

## Goal

Add `DeepSeek-OCR` to the existing local OCR comparison experiment as a raw-only mode so it can be compared side by side with `GLM-OCR`, `PaddleOCR-VL`, `PaddleOCR-VL-1.5`, and `FireRed-OCR`.

## Scope

This change stays intentionally narrow:

- Add one new mode: `deepseek_ocr`
- Run it as raw OCR only
- Do not add a parser-backed variant yet
- Reuse the current raw-only report so comparisons stay consistent

## Why This Approach

The main question for this round is whether `DeepSeek-OCR` is worth keeping in the local benchmark at all, and whether its official local inference path can expose any native result artifacts such as markdown output, bbox JSON, or native visualization images.

That means the first version should:

1. use the official inference path
2. preserve the raw output exactly
3. capture any native artifacts if the model writes them
4. avoid adding prompt variants or parser stages until the base path is proven useful

## Proposed Flow

For the new mode:

1. Load the model and tokenizer from the official Hugging Face model
2. Run the official `infer` path with the prompt `"<image>\nFree OCR."`
3. Save all official model-side artifacts into the current output directory under a dedicated native subdirectory
4. Extract the primary raw output into `raw_model_output.txt`
5. If the official inference path writes bbox or visualization files, expose them through:
   - `native_boxes`
   - `native_overlay_images`
6. Keep the current raw-only report contract:
   - raw output shown directly
   - native bbox shown only if actually available
   - otherwise clearly mark that no native bbox was returned

## Trade-offs

### Recommended: one raw-only DeepSeek mode

Pros:
- Smallest change
- Matches the current raw-only comparison need
- Makes the output easier to interpret

Cons:
- Does not explore alternate prompts yet

### Alternative: add both OCR and markdown prompt variants immediately

Pros:
- More coverage in one pass

Cons:
- Harder to compare because prompt behavior changes at the same time as model behavior
- Slower to run

This is intentionally deferred.

## Output Expectations

The output directory for `deepseek_ocr` should still contain the normal experiment files:

- `raw_model_output.txt`
- `run_summary.json`
- optional standard files already produced by the pipeline

In addition, DeepSeek-native artifacts should be written to a dedicated subdirectory inside that mode output folder. If the official inference path writes markdown, JSON, or visualization images, those files should remain intact there.

## Error Handling

If DeepSeek inference fails:

- the run should still complete with an empty envelope fallback
- the error should be recorded in `errors`
- the compare report should still show the failed mode

If DeepSeek returns raw text but no native bbox:

- the raw report should show the raw text
- native bbox should be marked as unavailable
- no synthetic bbox should be added

## Success Criteria

This change is successful if:

- `deepseek_ocr` can be run locally on the current invoice test images
- its raw output is readable in the compare report
- any native bbox or visualization outputs are preserved when available
- raw-only reports remain explicit that no parser-based overlay is being used
