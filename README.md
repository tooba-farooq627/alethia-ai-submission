# FineVideo second-by-second object indexing

Completed the first 50 pinned FineVideo shards: 1,650 videos, 461,757 output
seconds, and zero video errors. Full video coverage, record structure, and all
checkpoint/result hashes passed the final audit locally. The submitted output is
`results.jsonl.gz`; its provenance and measured session timings are in
`full-run-manifest.json` and `HARDWARE.md`. Implementation history is in
`ITERATIONS.md`.

Local and Colab checks: 11 correctness tests pass on both the Mac and a Colab
Python 3.11/Tesla T4 environment. A tiny real-model fixture produces matching
single-frame and batched results and handles a corrupt video on CPU and CUDA.
All 33 current tests pass locally and in the final Kaggle Linux runtime. This
includes a regression for first-snapshot creation on a newly booted host. Test discovery is limited
to `tests/`, so preserved implementation archives are not collected as tests.
The local suite includes output/coverage audits, memory-exhaustion retries that
retain every frame in order, checkpoint recovery, and submission finalization.
These checks and reference comparisons do not establish unseen-grader accuracy
or performance parity with its RTX 4090 environment.

## Install (grading machine: Ubuntu 22.04, Python 3.11, CUDA-capable driver)

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg libgl1 libglib2.0-0
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The local Mac environment is for development and small correctness checks.
The full dataset should run on an NVIDIA GPU. No Docker is required.

## Preparation (internet allowed, outside the timed run)

```bash
python prepare.py
```

This downloads the official `yolo11l.pt` to `models/yolo11l.pt`. That file must
remain present for the offline run. Model weights are not committed to Git.
The runner sets `YOLO_OFFLINE=true` before importing Ultralytics to disable its
online checks and synchronization during inference.
Preparation and inference select Matplotlib's `Agg` backend so an inherited
Colab inline-backend setting does not require notebook packages in the isolated environment.

## Required invocation (offline)

```bash
python run.py --input finevideo --output results.jsonl
```

The script recursively reads `.parquet` files. It never downloads data or weights
during this command. It evaluates every decoded frame with input size 640,
confidence 0.4, class-aware NMS at IoU 0.7, `max_det=300`, rectangular letterboxing,
and automatic precision (FP16 on CUDA, FP32 on CPU). Use `--precision fp32` for
the conservative reference. All COCO classes survive standard NMS before the five target
classes are selected. BGR frames are rotated using display metadata.

Presentation timestamps are normalized to the first decoded frame using rational
arithmetic. If a frame has no timestamp, a valid reported average frame rate is
used to reconstruct its time; otherwise that video receives an error. The first
video stream is processed. Audio and the `json` metadata column are unused.

Each second uses at-least-half frame voting and the floored median of per-frame
people counts. Empty seconds between timestamped frames are emitted. A decode
failure preserves completed seconds and adds an error record; the incomplete
current second is discarded. An unreadable Parquet footer/schema is reported to
stderr and causes exit code 2, since row identities cannot be recovered safely.

Output must be a new path: existing results are never overwritten silently.

## Validation and measurements

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python run.py --input sample --output artifacts/reference.jsonl --batch-size 1 --precision fp32 --profile artifacts/reference-profile.json
python run.py --input sample --output artifacts/batched.jsonl --batch-size 8 --precision auto --profile artifacts/batched-profile.json
python scripts/compare_results.py --reference artifacts/reference.jsonl --candidate artifacts/batched.jsonl
python scripts/capture_hardware.py --output artifacts/hardware.json
```

After a complete checkpointed run, audit its compressed results against the
manifest (substitute the actual checkpoint directory):

```bash
python scripts/audit_results.py --results checkpoints/results.jsonl.gz --manifest checkpoints/manifest.json
```

This checks record fields, continuous seconds, duplicate video groups, terminal
error records, all expected video identifiers, and counts against the manifest.
It does not measure detector accuracy or prove that the last second is complete.

For a small real-model integration check after preparation, run
`python scripts/smoke_test.py`. It downloads an official example image and creates
a three-frame MP4 plus a corrupt-video row in `artifacts/smoke/`. It intentionally
requires fresh output paths and is separate from submitted dataset results.
Use `--device 0 --output-dir artifacts/gpu-smoke` for a separate CUDA run.
`python scripts/gpu_check.py` checks CUDA inference on an official sample image.

Use a real, representative `sample` folder containing selected Parquet inputs.
The comparison tool reports diagnostic object micro-F1; the brief does not specify
the evaluator's averaging convention. It also checks missing/extra seconds and
error coverage. It does not replace the evaluator's unseen-video reference.

Stage timings are coarse wall times, not GPU kernel profiles. Process startup,
imports, and first-call model initialization also contribute to the official
external timer. `--batch-size 1 --precision fp32` supplies a conservative all-frame
reference. The required CLI defaults to the validated batch size of 8. CUDA out-of-memory failures
split the batch and retry the same frames; singleton failures still propagate.
GPU optimization validation status is recorded in `ITERATIONS.md`.

`scripts/optimize_and_run.py` benchmarks GPU precision/batching, then checks three
and twelve complete videos before a full run. Its object micro-F1/people exact/
within-one gates are 0.995/0.99/0.999 with zero sample errors and full second
coverage. It writes status and logs to the chosen persistent directory and stops
on any failed gate. Frame sampling is used only for the throughput probe;
validation and submission processing use every frame.

For the validated T4 configuration (FP16, batch 8), the recovery runner saves a
verified checkpoint after each complete video. Use an authorized persistent
directory and the pinned dataset revision:

```bash
python scripts/resumable_run.py --checkpoints /content/drive/MyDrive/Alethia_AI/first_full_resumable_v2 --revision 84c74091e1c6ee7a5dffabfafb5c9033e4718883
```

Rerun that command with identical source/configuration after restoring a CUDA
runtime. Completed video files are hash checked and reused; an interrupted
video is recomputed. Local compression followed by an atomic copy avoids
writing every second directly to Drive. The final output receives the full
50-shard coverage audit. This recovery command supplements the required
`run.py` submission CLI.

## Dataset access

Accept FineVideo's terms yourself at
https://huggingface.co/datasets/HuggingFaceFV/finevideo and authenticate securely:

```bash
huggingface-cli login
python scripts/download_data.py --output finevideo --count 1
```

For all required shards, use `--count 50` (about 25 GB). Never put an access token
in source, notebook output, shell history, or the repository. The Colab notebook
uses a hidden prompt if authentication is needed.
Use `--revision <dataset-commit>` to reproduce the same data across sample and
full runs. The notebook pins the revision before its first download and reuses it.

## Colab

`Colab_Setup.ipynb` contains a self-contained copy of project source so a public
repository is not needed to bootstrap. Upload/open it in Colab, select a GPU
runtime, then run setup and tests before downloading real data. It is generated
by `scripts/build_colab.py`; regenerate after source changes. Mounting Drive or
authorizing additional access is a separate user-controlled step.

## Kaggle recovery

After Colab GPU usage limits were reached, 60 completed video checkpoints and
one completed shard were recovered and hash verified locally. Build a private
Kaggle notebook from that checkpoint directory:

```bash
python scripts/build_kaggle.py --checkpoint /path/to/first_full_resumable_v2 --output Kaggle_Resume.ipynb --parallel
```

Select T4 x2 and enable Internet. The notebook embeds only an explicit source
and checkpoint allowlist; it does not embed tokens, chat exports, model weights,
or original videos. Enter the limited dataset token only in its hidden prompt.

The optional `scripts/parallel_run.py` assigns disjoint shards to separate GPU
processes, retaining the validated FP16/batch-8/every-frame pipeline. One process
writes the manifest and verifies worker output hashes. It accepts the original
v2 checkpoint signature and records the new coordinator's source hash. Local
recovery/corruption checks pass. The dual-GPU notebook first compares both GPUs
against the original three complete Colab video results, with strict agreement,
coverage, and error gates before launching the full run. Both T4s exactly matched
the saved Colab FP16 reference on all 1,082 validation seconds, with no missing,
extra, or error seconds. The final full-50 coverage audit passed; actual session
hardware and timings are recorded in `HARDWARE.md`.

The coordinator stops at a video boundary after its default ten-hour session
budget. Download the recovery ZIP and resume remaining work in a new session.
Do not assume an interactive Kaggle session or `/tmp` survives disconnection.
Only the complete, audited first 50 shards can be finalized for submission.

For unattended execution, use `--parallel --background` when building the
notebook, attach the separately authorized private Kaggle Secret, and select
**Save Version > Save & Run All**. Interactive sessions can stop on inactivity.
For a larger recovery, `--bundle-input /path/to/recovery.bin` writes the verified
source/checkpoint bundle separately; upload and attach it as a private input
before saving the run. The notebook verifies its SHA-256 before restoration.

After downloading a recovery ZIP through Kaggle's output-file menu, preserve it
and prepare a notebook containing the newest verified checkpoints:

```bash
python scripts/preserve_checkpoint.py --archive /path/to/alethia-recovery.zip --prepare-notebook
```

This checks ZIP integrity, the original source/model signature, every committed
video and shard hash, and the final coverage audit when all 50 shards are marked
complete. It keeps a new local snapshot rather than overwriting earlier evidence.

## Submission and provenance

The saved implementation and available validation evidence are in
`checkpoints/2026-11-20-validated-fp16/`, with file checksums and source commit
`f6dde75` (tag `validated-fp16-2026-11-20`). That local archive includes sample
evidence; it is not a completed FineVideo result file.

Check the installed CLI without Python network access using a previously saved
real-model fixture and reference:

```bash
python scripts/check_offline.py --input artifacts/optimized-cpu-smoke/input --reference artifacts/optimized-cpu-smoke/batch-3.jsonl --output-dir artifacts/offline-check-new
```

This hook denies Python DNS/connect operations and verifies itself before running
the CLI. It is not OS-wide isolation or a substitute for a fresh Linux/CUDA test.

After all 50 shards finish, copy their audited results and measured hardware facts
into the submission directory. This command rejects partial or modified results
and preserves any previous hardware record:

```bash
python scripts/finalize_results.py --run-dir /path/to/completed/checkpoints --output-dir .
python scripts/export_ai_chat.py
python scripts/check_submission.py --manifest full-run-manifest.json --output artifacts/submission-readiness.json
```

The chat exporter runs in this local Codex session. It preserves visible messages
and tool records and replaces recognizable credential strings with `[REMOVED]`.
Refresh it just before submission; screenshots and other tool histories also need
to be checked for credentials before public publication. The readiness command
checks only that `WRITEUP.md` exists and is nonempty; it never reads its contents.

The completed `results.jsonl.gz` and actual full-run hardware/timings are included
in this repository. The author must complete `WRITEUP.md` independently as the
brief requires. Include histories from any other AI tools used, refresh the Codex
export before submission, and verify public repository visibility before sending
the repository link.

This repository starts from a fresh snapshot of the current project files,
without the earlier repository's Git history. The `ai_chats/` directory contains
an overview, not the complete history requested by the brief. This snapshot
should not be treated as meeting the full AI-history requirement.
The initial FP32 full run did not finish, so no completed full-run baseline
timing is claimed. Do not publish the dataset or tokens.
