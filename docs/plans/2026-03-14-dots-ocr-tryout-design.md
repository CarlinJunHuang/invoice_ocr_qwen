# dots.ocr 1.5 Tryout Design

## Goal

Create a separate, minimal tryout area for `dots.ocr 1.5` so the model can be tested without polluting the existing OCR comparison experiment.

## Scope

This tryout is intentionally narrower than the main OCR experiment:

- Test only raw model output
- Do not parse invoice fields
- Do not build bbox overlays
- Do not integrate with the existing compare pipeline yet
- Keep `HF local inference` and `vLLM + docker` isolated in one small folder

## Recommended Approach

Use a dedicated folder named `dots_ocr_tryout` inside the repo and run the model in two stages:

1. try `Hugging Face local inference`
2. if that fails or produces unusable output, try `vLLM + docker`

This keeps the model evaluation focused on the actual blocker: whether the publicly available `dots.ocr 1.5` weights can produce readable raw OCR output locally.

## Why Isolation Matters

The main OCR experiment already contains compatibility workarounds for multiple document models. `dots.ocr 1.5` has a different distribution path and a less stable public setup, so it should not be mixed into the main experiment until it proves it can run reliably.

## Output Contract

Each run should only preserve:

- the exact command or config used
- the raw stdout / model output
- a short status file recording success or failure
- optional logs from HF or vLLM startup

No field extraction or downstream post-processing should happen in this tryout.

## Execution Order

### Stage 1: Hugging Face local inference

Use the community-accessible model source first.

Success means:

- the model runs locally
- the output is readable text or markdown
- the output is not just the prompt echoed back

Failure means:

- model load/runtime errors
- empty output
- prompt echo
- obvious gibberish

### Stage 2: vLLM + docker

Only attempt this if Stage 1 fails.

Success means the same thing: readable raw OCR output, preserved exactly as returned.

## Success Criteria

This tryout is successful if it ends with one of these clear outcomes:

1. `dots.ocr 1.5` works with HF local inference and produces readable raw output
2. HF does not work, but `vLLM + docker` does
3. neither path works cleanly, and the failure mode is documented clearly enough to stop further effort
