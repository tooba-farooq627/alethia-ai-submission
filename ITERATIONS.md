# Execution iterations

## 1. Setup and exact baseline (local and Colab GPU checks passed)

- Python 3.11 local environment, exact dependencies, Git repository.
- Required command-line interface and an all-frame YOLO11l pipeline.
- Exact timestamp bins, presence voting, median counts, rotation, error records.
- Automated semantic tests plus a real-model smoke run.
- Colab notebook and GPU availability check using the user's personal account.
- Dataset access requires the user's Hugging Face account and acceptance of its terms.

Verified locally on 20 November 2026:

- Python 3.11.16 environment installed; exact dependency lock resolves cleanly.
- All 11 semantic/decoder tests pass, including display rotation against FFmpeg.
- Official YOLO11l weights load successfully.
- Real-model smoke fixture: 3 frames, 2 second records, plus 1 corrupt-video error.
- Single-frame and batch-of-3 outputs match on that tiny fixture (diagnostic micro-F1
  1.0 and exact people agreement 1.0). This is not FineVideo validation evidence.
- The real-model smoke run also passes with outbound socket connections disabled.
- Colab: Python 3.11.14, Torch 2.5.1+cu124, Tesla T4 with 15,360 MiB GPU memory.
- All 11 correctness tests also pass in Colab. Official weights load there.
- Real GPU inference and the end-to-end GPU smoke fixture pass. Single-frame
  and batched counts match on the fixture; this is not full FineVideo validation.
- Fixed the inherited Colab inline plotting backend by selecting headless Agg
  before importing Ultralytics. Preparation and inference pass with the original
  incompatible backend present in the parent environment.
- Notebook setup now reuses its Python environment and existing system packages,
  captures error output, and keeps a newly entered HF token in runtime memory.
- Hugging Face sign-in is complete. After the user's explicit approval, FineVideo
  terms were accepted and the site confirmed dataset access was granted.
- With explicit approval, a FineVideo-only read token was created and entered into
  Colab's hidden prompt. Authentication as `tooba25` succeeded; the token remains
  in runtime memory and is absent from notebook source and repository files.
- Downloaded the 493,919,884-byte first shard at dataset commit
  `84c74091e1c6ee7a5dffabfafb5c9033e4718883`. The three-video sample contains
  29,782 frames and about 1,080 seconds of video. Single-frame and batch-8
  all-frame validation is running; results are not yet claimed.
- User explicitly approved connecting this notebook to personal Google Drive.
  The mount is verified, and setup plus current artifacts are saved in
  `My Drive/Alethia_AI`. The full run will checkpoint there.
- Real-data validation completed: all 29,782 frames from three videos produced
  1,082 second records in each run, with no errors or missing/extra seconds.
  Diagnostic object micro-F1, people exact agreement, and within-one agreement
  are all 1.0 against the single-frame reference. This is subset agreement,
  not evaluator accuracy on unseen videos.
- Colab sample wall times: batch 1 = 722.746596597 seconds; batch 8 =
  688.931001214 seconds (about 43.23 frames/s). Sample video duration is about
  1,079.84 seconds. Duration-based extrapolation to the brief's approximately
  125 hours is about 79.75 hours, excluding downtime and download overhead;
  the estimate depends on the rest of the dataset resembling this small sample.
- After validation passed, evidence was copied to Drive and the 50-shard baseline
  started in `Alethia_AI/first_full_baseline_v1`. The start was observed in Colab;
  no completed-shard count or full-run runtime has yet been verified.
- Added a streaming output audit for record validity and manifest-based coverage.
  All 19 local tests pass, including rejection of gaps, duplicate seconds,
  repeated video groups, invalid counts/classes, and missing full-run videos.

Exit: tests pass, the actual model processes a small input, and a GPU session is ready.

## 2. Real-data validation and baseline measurement

- Inspect the actual first-shard schema and a few representative videos.
- Compare single-frame and batched all-frame outputs on the same videos.
- Record input revision, code revision, hardware, errors, and coarse stage timings.
- Run the first working version over all 50 shards; preserve total wall time.
- Save output checkpoints outside ephemeral Colab storage.

Exit: real-data correctness evidence and first full-run results/timing exist.

## 3. Measured optimization

Started on 20 November 2026:

- Stopped the incomplete FP32 full run because its sample-based projection was
  about 80 hours. Its existing files are preserved; there is no completed FP32
  full-run timing to report.
- Actual FP32 sample profile: inference 648.84 seconds, decoding 36.19 seconds;
  inference accounts for about 95% of measured processing time.
- Added CUDA FP16 selection, an explicit FP32 reference option, and ordered batch
  splitting on CUDA memory exhaustion. Every submission/validation frame remains
  processed; only the small throughput probe samples frames.
- All 21 local tests pass. The real CPU model smoke run also passes with automatic
  FP32 fallback and matching single/batched outputs.
- The GPU throughput probe selected FP16 batch 8 on this T4. Its complete
  three-video run took 300.1592 seconds versus FP32 batch 8 at 688.9310 seconds
  (about 2.30x faster, 99.22 frames/s). Projection: about 34.75 continuous hours
  for similar data, excluding downtime and download overhead.
- Three-video FP16 agreement: object micro-F1 0.9992831541, people exact
  0.9981515712, people within one 1.0, zero errors/missing/extra seconds. The
  outputs are not identical; they pass gates tighter than the assignment's.
- The unattended worker required object
  F1 >= 0.995, exact people >= 0.99, within-one >= 0.999, full second coverage,
  and zero sample errors before starting all 50 shards. A failed gate stops and
  saves its reason. Its full-run target is `Alethia_AI/first_full_optimized_v1`.
- Twelve-video validation passed: 3,692 seconds, object micro-F1 0.9996488764,
  people exact 0.9994582882, within-one 1.0, zero errors/missing/extra seconds.
- On return, the previous runtime was gone. The original optimized manifest
  contained zero completed shards. An unfinished output had the first three
  videos, but was not a reusable completed shard. Its files are preserved.
- Restored the same GPU environment and validated inference source from Drive.
  Added `scripts/resumable_run.py` to save each complete video atomically with
  a hash; inference/decoding/aggregation still use the validated functions.
  Compression now occurs on local disk before a single copy to Drive.
- All 22 local tests pass, including interruption/reuse/corruption coverage;
  the restored runtime's available 12 tests also pass. The new source archive
  was CRC checked and saved to Drive as `resumable-source.zip`.
- Restarted into `Alethia_AI/first_full_resumable_v2`. Its first durable video
  checkpoint was observed at 2026-11-20 05:49:54 UTC, with zero errors.
  The first-video projection was about 30 hours remaining and is preliminary.
- The resumed runner reused its first saved video (manifest update 05:53:49 UTC).
  Browser control then became unavailable with `cgWindowNotFound`, preventing
  verification of further progress. Current worker activity must be checked;
  a historical running label is not evidence that the worker remains active.

- Profile decoding, inference, data movement, and writing.
- Tune batching and decode concurrency against representative baseline outputs.
- Add bounded prefetch only if decoding starves the GPU.
- Treat reduced precision, model export, and sampling as experiments requiring validation.
- Retain the mandated weights, preprocessing, thresholds, and postprocessing.

Exit: a measured improvement passes all correctness checks; no speed claim without measurements.

## Saved checkpoint and submission preparation (20 November 2026)

- Preserved source commit `f6dde75` as tag `validated-fp16-2026-11-20` and
  `checkpoints/2026-11-20-validated-fp16/implementation.zip`. The directory also
  contains the available validation reports, actual CPU fixture outputs, recovery
  evidence, both PDFs, snapshot provenance, and 19 verified file checksums.
  GPU raw reference/candidate outputs remain on Drive; they were not recovered
  locally because its app connection still requires authentication.
- Aligned the required `run.py` default with the validated batch size of 8.
  This does not change the frozen per-video runner source/config signature.
- Added a reproducible offline CLI check with a self-tested Python socket/DNS
  guard. On the Mac it ran all three fixture frames, made zero network attempts,
  preserved the corrupt-video error, and matched the saved reference exactly.
  This is not an RTX 4090 or fresh Ubuntu VM benchmark.
- Added finalization that verifies result hashes and complete first-50 coverage
  before copying submission results and recording actual hardware/timing data.
  Incomplete runs are rejected before submission files are created.
- Exported this Codex session's visible messages/tool records without summaries
  or shortened tool content. Its provenance/cutoff is in `ai_chats/export-manifest.json`.
  Refresh before submission; no claim of completeness for later activity or other tools.
- Added a local submission readiness report. Current missing items are full-50
  results/timings, a configured public repository, and the user's own write-up.
  The readiness tool checks write-up existence/size only, never its contents.
- All 25 local tests pass. The clean Colab notebook was regenerated and all 31
  embedded source files were checked byte-for-byte against the working source.
- A separate Mac environment install was attempted, but stopped during slow
  dependency downloads. It is not counted as a passed fresh-install check.
- Restored browser access revealed saved recovery output for six complete videos,
  1,576 seconds and zero errors at 06:00:20 UTC. The session was later disconnected.
  A reconnect succeeded, but successful processing resumption was not verified.
- The complete current visible chat was also saved privately under
  `artifacts/private-chat-export/`. Public chat exports still need refresh and
  review before publication; no public repository has been created or pushed.

## 4. Final run and reproducible submission

### Recovered data and Kaggle preparation (20 November 2026)

- Downloaded `alethia-checkpoint-backup-20261120T093340Z.zip` and verified its ZIP
  integrity and all 60 video checkpoint hashes locally. The manifest has one
  completed shard, 16,412 output seconds and zero errors, updated 08:34:23 UTC.
  Its frozen source/model/config signature still matches the working tree.
- The recovered archive also contains the original GPU validation outputs and
  profiles under `optimization_v1/` and `artifacts/`, superseding the earlier
  note that these could only be accessed on Drive.
- Colab GPU usage limits stopped further GPU access. The user selected Kaggle
  and authorized uploading project/checkpoint files and using the limited
  FineVideo token in Kaggle runtime memory. No token persistence was authorized.
- Verified account phone/identity checks and 30 unused GPU hours; selected T4 x2.
  Imported the single-GPU recovery notebook. Runtime startup was observed, but
  checkpoint restoration and inference were not verified.
- Prepared a dual-GPU coordinator with disjoint shard ownership, one manifest
  writer, original checkpoint compatibility, per-video hashes and a ten-hour
  session budget. All 31 local and Kaggle Linux tests pass.
- Mac browser control remained unreliable; the user completed personal-account
  sign-in in the built-in browser. Confirmed private notebook visibility and
  enabled Files only persistence, without persisting runtime variables/tokens.
- Both T4 GPUs processed all 29,782 frames of the original three complete videos
  and exactly matched the saved Colab FP16 output (1,082 seconds per GPU, zero
  errors/missing/extra seconds). Concurrent validation completed at 10:37:29 UTC;
  both profiles and the report were recovered locally and hash checked.
- Started the full two-GPU coordinator at 10:39:04 UTC. Its live monitor reported
  63 saved videos, one complete shard and zero errors at 10:42:53 UTC, with both
  GPUs active. This remains an incomplete full run.
- Added a continuous notebook monitor and a verified recovery ZIP refreshed
  every ten minutes. Snapshots retain committed checkpoint members, migration
  outputs/profiles, GPU logs and an explicit source allowlist. The first archive
  was downloaded to `checkpoints/kaggle/20261120T104225Z/` and its CRC and all
  60 included video hashes were independently verified on the Mac.

- Full final run; validate per-video coverage and error records; gzip the results.
- Record actual hardware and both full-run timings.
- Fresh-install and offline-runtime checks on Linux/CUDA.
- Verify public repository contents and preserve complete AI chat exports.
- User writes WRITEUP.md independently. Do not use AI to write, rewrite, or check answers.

Exit: all required files exist with genuine run evidence and are committed before the deadline.

## Current limitations

The grading machine's RTX 4090 and fresh Ubuntu installation have not been
benchmarked. The initial FP32 full run did not complete, so no completed initial
full-run baseline time is available. Reference agreement is a diagnostic rather
than an unseen-grader accuracy result. The independent write-up, public
repository publication, and final chat privacy/completeness review remain.

## Completed first-50 run (21 November 2026)

- The user explicitly authorized persistent storage of the limited FineVideo
  token as a private Kaggle Secret for this notebook's unattended background
  sessions. Its value is absent from notebook source and result archives.
- The first interactive Kaggle session stopped on inactivity. Switched to
  Save & Run All; its ten-hour chunk saved 1,336 videos and 40 complete shards
  with zero video errors. Every saved hash was independently verified locally.
- The next saved version passed the both-GPU comparison but failed before new
  processing: a fresh host's monotonic clock was below ten minutes, so the
  monitor skipped its initial snapshot and tried to stat a missing ZIP.
  Reproduced that failure in a regression test, fixed immediate first-snapshot
  creation in `c28d1c4`, and passed all 33 tests locally and on Kaggle.
- Corrected saved version 3 (`372131625`) restored the verified checkpoint and
  passed exact agreement on both T4s over 1,082 reference seconds each, with
  zero missing/extra/error seconds. The first snapshot and subsequent new-video
  progress were observed before calling the restart healthy.
- Full processing finished at 2026-11-21 12:42:41 UTC. The final monitor reported
  `complete_audited` at 12:43:04 UTC: 50 shards, 1,650 videos, 461,757 output
  seconds, zero video errors, and full video coverage checked.
- Downloaded and preserved the final 5,379,412-byte recovery archive under
  `checkpoints/kaggle/20261121T141913.065855Z/`. Its SHA-256 is
  `85ee3676faff5408be4fbb7764eeed9b91d54b2a8665a280ba83ea64501993f1`.
  ZIP integrity, all video/shard hashes, the final result hash, and full-50
  coverage were verified locally before finalization.
- Finalized `results.jsonl.gz`, `HARDWARE.md`, `full-run-manifest.json`, and
  `final-audit.json`. Results SHA-256:
  `6f563e4a71b6855c4aacf1f9af1b390e9f8928c5568a974326fdf20c2b9870b2`.
  Hardware records distinguish recorded active session time from elapsed time
  across interruptions, and do not invent a completed FP32 baseline runtime.
