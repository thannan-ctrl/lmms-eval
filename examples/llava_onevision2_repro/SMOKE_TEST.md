# Single-video smoke test

Sanity check + frames-vs-codec latency comparison for `llava_onevision2`
on one EgoSchema video and one Video-MME video, run in the exact Docker
environment from the top-level `README.md` § 1. Script:
`run_single_video.py`.

## Results (2026-09-09, A100x2)

Videos: EgoSchema `0074f737-11cb-497d-8d07-77c3a8127391` (180s) and
Video-MME `fFjv93ACGo8` question `001-1` (74s).

Frames split into `fetch_video` (decord decode+resize) vs.
`image_processor` (PIL frames → tensors); codec split into `transcode`
(mpeg4→H264, see below) vs. `cv_preinfer` (the CLI subprocess) vs. its
own `image_processor` (canvas JPEGs → tensors) vs. `other` (padding
drop + position calc + tokenize, not split further):

| Sample | Dur | Backend | Tokens | Transcode | fetch_video | cv_preinfer | image_processor | other | VLM | **E2E** | E2E−transcode | vs. frames(64f) | vs. frames(64f), no transcode |
|---|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| egoschema | 180s | frames (64f, token-matched) | 12288 | 0.00s | 0.22s | – | 0.10s | – | 3.74s | **4.06s** | 4.06s | 1.00x | 1.00x |
| egoschema | 180s | codec | 12288 | 6.15s | – | 0.49s | 0.06s | 0.59s | 4.06s | **11.36s** | 5.21s | 2.80x | 1.28x |
| videomme | 74s | frames (64f, closest match) | 9216 | 0.00s | 0.18s | – | 0.07s | – | 3.11s | **3.36s** | 3.36s | 1.00x | 1.00x |
| videomme | 74s | codec | 11520 | 1.66s | – | 0.38s | 0.05s | 0.53s | 4.58s | **7.21s** | 5.55s | 2.15x | 1.65x |

![latency breakdown](latency_breakdown.png)

**Key findings** (n=1/cell, illustrative not a benchmark):
- **Transcode dominates codec's overhead** (54%/23% of E2E). Artifact of
  local `mpeg4` videos, not H264/HEVC (`cv-preinfer`'s requirement) —
  zero on a native-H264/HEVC corpus. Scales with total pixels, not
  duration (2.4x longer video → 3.7x slower transcode, since also taller).
- **Codec is still 1.3–1.65x slower even without transcode**, at matched
  token budgets — VLM itself runs slower per token, though codec's own
  `image_processor` is cheaper than frames'.
- **`fetch_video` + both `image_processor`s scale with frame/canvas
  count, not duration** — fixed at 64 here, and decord seeks straight to
  sampled indices rather than decoding the whole file.
- **`cv_preinfer` looks duration-independent here but isn't in general**
  (varies a lot across a larger EgoSchema sample, not shown here). It
  samples a fixed `512`-frame candidate pool
  (`(target_canvas//images_per_group)*group_size`), clamped down only
  for videos shorter than 512 frames (~17s @30fps) — neither test video
  hit that clamp. So the *target count* is length-independent, but
  decoding those 512 timestamps still costs more on a longer/higher-res
  video.
- **`--num-frames` is a no-op for codec** — codec's size comes entirely
  from `--codec-target-canvas` (a canvas count, not a frame count); both
  set to `64` here just to roughly match token counts.

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
