# Single-video smoke test

Sanity check + frames-vs-codec latency comparison for `llava_onevision2`
on one EgoSchema video and one Video-MME video, run in the exact Docker
environment from the top-level `README.md` § 1. Script:
`run_single_video.py`.

## Results (2026-09-09, A100x2)

Videos: EgoSchema `0074f737-11cb-497d-8d07-77c3a8127391` (180s) and
Video-MME `fFjv93ACGo8` question `001-1` (74s).

| Sample | Dur | Backend | Tokens | Transcode | Preprocess | VLM | **E2E** | E2E−transcode | vs. frames(64f) | vs. frames(64f), no transcode |
|---|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|
| egoschema | 180s | frames (64f, token-matched) | 12288 | 0.00s | 0.59s | 3.78s | **4.37s** | 4.37s | 1.00x | 1.00x |
| egoschema | 180s | codec | 12288 | 6.25s | 0.77s | 4.00s | **11.02s** | 4.77s | 2.52x | 1.09x |
| videomme | 74s | frames (64f, closest match) | 9216 | 0.00s | 0.26s | 3.19s | **3.45s** | 3.45s | 1.00x | 1.00x |
| videomme | 74s | codec | 11520 | 1.68s | 0.73s | 4.47s | **6.88s** | 5.20s | 1.99x | 1.51x |

**Summary**: transcode is codec's single biggest overhead — 57% of E2E
for EgoSchema (6.25s/11.02s), 24% for Video-MME (1.68s/6.88s) — and
exists *only* because these videos are `mpeg4`, not H264/HEVC (which
`cv-preinfer` requires); on an H264/HEVC-native corpus it's zero. It's
CPU-bound (`ffmpeg -preset fast`) and scales with total pixels encoded
(frames × width × height), not just duration: EgoSchema is 2.42x
longer than Video-MME (180s vs. 74.3s) but also taller (448×336 vs.
448×252, 1.33x), so it has 3.23x more total pixels — closely tracking
the observed 3.72x transcode-time ratio (6.25s vs. 1.68s), with the
remaining gap consistent with run-to-run noise. A faster preset or
hardware encoding would shrink transcode further either way. Even with
transcode removed, codec is still 1.1–1.5x slower than frames at
matched token budgets — VLM runs consistently slower per comparable
token, while canvas packing is comparable to frames' video decode.
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
