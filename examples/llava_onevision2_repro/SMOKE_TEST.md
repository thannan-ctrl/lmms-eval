# Single-video smoke test

Sanity check + frames-vs-codec latency comparison for `llava_onevision2`
on one EgoSchema video and one Video-MME video, bypassing the registered
`lmms_eval` tasks (their expected dataset layout/HF cache doesn't match
the local mirror) and running in the exact Docker environment from the
top-level `README.md` § 1.

- Script: `run_single_video.py` — loads the model once, runs each sample
  through `--backends` (default `frames,codec`), prints a latency table.
- Data: `<DATA_ROOT>/egoschema/subset.json` + `.../video_mme/questions.json`,
  matched against local video files (first entry with a matching file per
  dataset). Default `DATA_ROOT`: `/home/thannan/scratch/AutoGaze/data`.
- Prompt: MCQ with brief grounded reasoning before `Answer: X`, parsed and
  compared against ground truth.

## Reproduce

Hold a GPU allocation (`--jobid` reuse avoids re-queueing per command):

```bash
salloc --partition=A100x2 --nodes=1 --gres=gpu:1 --time=08:00:00 sleep 28800 &
JOBID=<id from squeue -u $USER>
```

Then, per run:

```bash
srun --jobid=$JOBID bash -c '
    cd /home/scratch.thannan_wwfo/LLaVA-OneVision-2/lmms-eval
    docker build -t lmms-eval-ov2:latest -f dockerfile/Dockerfile .
    mkdir -p /home/thannan/scratch/hf_cache
    docker run --rm --gpus all --ipc=host --shm-size=16g --network=host \
      --user $(id -u):$(id -g) -e HOME=/tmp \
      -v $(pwd):/workspace/lmms-eval \
      -v /home/thannan/scratch/hf_cache:/hf_cache \
      -v /home/thannan/scratch/AutoGaze/data:/data \
      -e HF_HOME=/hf_cache -e LLAVA_CODEC_ONLINE_TUNED=1 \
      lmms-eval-ov2:latest bash -c "
        export PATH=/tmp/.local/bin:\$PATH
        cd /workspace/lmms-eval
        pip install --user -e . --no-deps -q
        python3 -m pip install --user -q \
            --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ \
            codec-video-prep-legacy-exact==0.2.5.post2
        python3 examples/llava_onevision2_repro/run_single_video.py \
            --data-root /data --num-frames 32 \
            --backends frames,codec --codec-target-canvas 64
      "
'
```

**Notes:**
- `A100x2` (x86_64), not `gb200nvl72_preprod`: that partition is
  `aarch64` (Grace CPU) and `decord` has no aarch64 PyPI wheels, so the
  Dockerfile's install fails there. `A100x2` also matches the README's
  "verified on 8 × A100-80GB" environment, and has `/home/scratch.thannan_wwfo`
  mounted (confirm with `ls` — not all partitions do).
- `--user $(id -u):$(id -g) -e HOME=/tmp`: NFS root-squashes `docker
  run`'s default root user, so it can't write into the mounted repo
  (`pip install -e .`) or HF cache. Run as the host user instead.
- `-e LLAVA_CODEC_ONLINE_TUNED=1`: the checkpoint's *default* codec path
  calls a `cv-preinfer` binary that only the (README-forbidden) regular
  `codec-video-prep` package provides. This repo's separate "online
  tuned" codec path correctly calls the pinned
  `codec-video-prep-legacy-exact` CLI instead — but it's opt-in via this
  env var.
- **Transcode step**: `cv-preinfer`'s bitcost/readiness scorer only
  accepts H264/HEVC bitstreams. Both local datasets are natively
  `mpeg4` (MPEG-4 part 2), so `run_single_video.py` transcodes a
  throwaway H264 copy (`ffmpeg -c:v libx264 -preset fast -crf 23`)
  before handing a video to the codec backend, timing it separately
  (`transcode_latency_s`). Irrelevant if your source videos are already
  H264/HEVC.

## Results (2026-09-09, A100x2)

| Sample | Backend | Predicted | GT | Transcode | Preprocess | VLM | **E2E** | vs. frames |
|---|---|:-:|:-:|--:|--:|--:|--:|--:|
| `egoschema/0074f737...` | frames | D | D ✅ | 0.00s | 0.38s | 2.59s | **2.97s** | 1.00x |
| `egoschema/0074f737...` | codec | D | D ✅ | 6.50s | 0.85s | 3.99s | **11.34s** | 3.82x |
| `videomme/fFjv93ACGo8/001-1` | frames | A | C ❌ | 0.00s | 0.12s | 1.64s | **1.76s** | 1.00x |
| `videomme/fFjv93ACGo8/001-1` | codec | A | C ❌ | 1.60s | 7.60s | 4.48s | **13.68s** | 7.76x |

Predictions are identical between backends (both match/miss ground truth
the same way); the codec backend is 3.8–7.8x slower end-to-end here,
driven mostly by preprocessing (canvas packing) and the transcode step,
plus a higher VLM cost since `codec-target-canvas=64` produces more
tokens than the 32-frame budget used for `frames` (not a matched token
budget — noted, not controlled for). n=1 per cell: illustrative
single-sample timings, not a throughput benchmark.

Both source videos here happen to be `mpeg4` (MPEG-4 part 2) — an older
codec, not what `cv-preinfer` requires — which is why the transcode step
exists at all. If the videos had originally been encoded/ingested as
H264 or HEVC (as most modern video corpora are), that 1.6–6.5s transcode
cost disappears entirely: the codec backend would read the source file
directly, so its E2E latency would drop to roughly `preprocess + vlm`
(e.g. ~4.8s instead of 11.3s for the EgoSchema sample). The transcode
overhead is purely an artifact of this dataset's source encoding, not an
inherent cost of the codec backend.
