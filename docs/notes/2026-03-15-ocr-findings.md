# OCR Findings

Date: 2026-03-15

## Scope

This note captures the current qualitative observations from the local invoice OCR experiments on the internal sample images `1a.png` and `2a.png`.

## Current observation

- `dots.ocr 1.5` produced the strongest raw layout output in this round.
- Its native bbox output was materially better than the other OCR models tested so far.
- On the same sample invoices, the result quality gap was large enough to be visible without parser-side post-processing.

## Practical implication

- For the current POC direction, `dots.ocr 1.5` is the most promising OCR front-end among the locally tested options.
- It is worth keeping the other models for baseline comparison, but `dots.ocr 1.5` should be treated as the current leading candidate for invoice OCR and layout extraction.

## Caveat

- This is still a small-sample observation, not a formal benchmark.
- The conclusion is based on internal qualitative review of raw outputs and native bbox overlays, not on a labeled evaluation set.
