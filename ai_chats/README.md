# AI assistance: technical overview

This page is an AI-generated summary of the engineering work, not an original
chat transcript. This is a fresh repository snapshot with no inherited Git
history. The JSONL chat export and its manifest are not included in this
repository.

The assignment brief asks for full AI chat histories. This overview alone does
not satisfy that requirement.

## Scope of assistance

Codex provided substantial assistance with implementation, test creation,
profiling and result comparisons, cloud notebook setup, checkpoint recovery,
debugging, documentation and repository preparation. The candidate supplied
the brief, requested explanations and deadline-driven optimization, chose
available compute options, and authorized account/data-access operations.
This was extensive AI assistance, rather than occasional autocomplete.

## Technical decisions and evidence

| Area | Work performed with AI assistance | Where to inspect it |
|---|---|---|
| Match the specification | Implemented presentation-timestamp bins, rotation, every-frame inference, at-least-half presence voting, floored median people counts and damaged-video handling. | [Decoder](../video_indexer/decode.py), [aggregator](../video_indexer/aggregate.py), [correctness tests](../tests/test_correctness.py) |
| Optimize without dropping frames | Compared reference and optimized runs; selected batch 8 / CUDA FP16; added smaller-batch memory retries that preserve all frames and their order. | [Pipeline](../video_indexer/pipeline.py), [memory-retry tests](../tests/test_memory_retry.py), [12-video comparison](../evidence/validated-fp16/optimization-validation-passed.json) |
| Recover long GPU runs | Moved from Colab to Kaggle; assigned disjoint shards to two GPUs with a single manifest writer; checked per-video hashes when resuming. | [Parallel coordinator](../scripts/parallel_run.py), [checkpoint tests](../tests/test_video_checkpoint.py), [GPU migration comparison](../evidence/kaggle-migration/validation-report.json) |
| Diagnose operational failures | An interactive session stopped on inactivity. A later background attempt failed because the monitor skipped the first recovery ZIP on a fresh host. The monitor failure was reproduced in a regression test and fixed. | [Execution history](../ITERATIONS.md), [monitor](../scripts/watch_run.py), [startup regression](../tests/test_watch_run.py) |
| Verify completed output | Checked full expected-video coverage, records and hashes before finalizing all-50 results. | [Final audit](../final-audit.json), [full-run manifest](../full-run-manifest.json), [hardware and timings](../HARDWARE.md) |

## Limits of the evidence

Reference agreement was measured on selected complete videos. Full-dataset
coverage and file-integrity auditing are separate checks; they are not an
unseen-grader accuracy score. The initial FP32 full run did not finish, so no
completed baseline full-run time is claimed. The grading machine's RTX 4090
was not benchmarked here.
