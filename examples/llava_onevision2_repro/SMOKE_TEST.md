# Single-video smoke test

Sanity check + frames-vs-codec latency comparison for `llava_onevision2`
on one EgoSchema video and one Video-MME video, run in the exact Docker
environment from the top-level `README.md` § 1. Script:
`run_single_video.py`.

## Results (2026-09-09, A100x2)

| Sample | Dur | Backend | Tokens | Predicted | GT | Transcode | Preprocess | VLM | **E2E** | vs. frames |
|---|--:|---|--:|:-:|:-:|--:|--:|--:|--:|--:|
| `egoschema/0074f737...` | 180s | frames | 6144 | D | D ✅ | 0.00s | 0.36s | 2.60s | **2.95s** | 1.00x |
| `egoschema/0074f737...` | 180s | codec | 12288 | D | D ✅ | 6.25s | 0.77s | 4.00s | **11.02s** | 3.73x |
| `videomme/fFjv93ACGo8/001-1` | 74s | frames | 4608 | A | C ❌ | 0.00s | 0.12s | 1.64s | **1.76s** | 1.00x |
| `videomme/fFjv93ACGo8/001-1` | 74s | codec | 11520 | A | C ❌ | 1.68s | 0.73s | 4.47s | **6.88s** | 3.90x |

Preprocess breakdown (all sub-ms/negligible stages omitted — `_build_messages` frame-decode dominates for `frames`, the processor call dominates for `codec`):

| Sample | Backend | build_messages | chat_template | processor call |
|---|---|--:|--:|--:|
| egoschema | frames | 0.30s | 0.007s | 0.05s |
| egoschema | codec | 0.00s | 0.000s | 0.77s (cv-preinfer + image proc) |
| videomme | frames | 0.09s | 0.000s | 0.03s |
| videomme | codec | 0.00s | 0.000s | 0.73s (cv-preinfer + image proc) |

**Summary**: predictions match between backends (same right/wrong per
sample). Codec uses ~2x the video tokens of frames here (target_canvas=64
vs. a 32-frame budget — not a matched token budget) and is 3.7–3.9x
slower end-to-end, split roughly evenly between the one-time transcode
and a heavier VLM pass; codec's own canvas-packing/image-processing step
(`processor call`) is actually comparable to or cheaper than frames'
raw video decode. `codec`'s E2E varied run-to-run (11.3s vs. 6.9s across
two runs of the same EgoSchema sample) — likely OS page-cache/CPU
contention noise in `cv-preinfer`'s subprocess, not a deterministic cost.
The transcode cost is an artifact of the local videos being `mpeg4`, not
H264/HEVC — with H264/HEVC-native sources it disappears entirely.
n=1/cell: illustrative, not a throughput benchmark.

<details>
<summary><b>Reproduce</b> (Docker env, persistent GPU allocation, notes)</summary>

Data: `<DATA_ROOT>/egoschema/subset.json` + `.../video_mme/questions.json`
matched against local video files. Default `DATA_ROOT`:
`/home/thannan/scratch/AutoGaze/data`. Prompt: MCQ + brief reasoning
before `Answer: X`, parsed vs. ground truth.

```bash
salloc --partition=A100x2 --nodes=1 --gres=gpu:1 --time=08:00:00 sleep 28800 &
JOBID=<id from squeue -u $USER>   # --jobid reuse avoids re-queueing per run

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

**Notes** (all cluster/data-specific, not README bugs):
- `A100x2`, not `gb200nvl72_preprod`: that partition is `aarch64` and
  `decord` has no aarch64 wheels. `A100x2` matches the README's
  "verified on 8 × A100-80GB" and has `/home/scratch.thannan_wwfo`
  mounted (confirm with `ls` — not all partitions do).
- `--user $(id -u):$(id -g) -e HOME=/tmp`: NFS root-squash blocks
  `docker run`'s default root user from writing the mounted repo/cache.
- `-e LLAVA_CODEC_ONLINE_TUNED=1`: the checkpoint's default codec path
  calls a `cv-preinfer` binary only the README-forbidden regular
  `codec-video-prep` package provides; this repo's "online tuned" path
  correctly calls the pinned `codec-video-prep-legacy-exact` CLI, opt-in
  via this env var.
- Transcode: `cv-preinfer` only accepts H264/HEVC; local videos are
  `mpeg4`, so `run_single_video.py` transcodes a throwaway H264 copy
  (`ffmpeg -c:v libx264 -preset fast -crf 23`) per video, timed
  separately as `transcode_latency_s`.

</details>
