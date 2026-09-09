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
`cv-preinfer` requires); on an H264/HEVC-native corpus it's zero. Even
with transcode removed, codec is still 1.1–1.5x slower than frames at
matched token budgets — VLM runs consistently slower per comparable
token, while canvas packing is comparable to frames' video decode.
n=1/cell: illustrative, not a throughput benchmark.

**Transcode scales with pixels, not duration**: EgoSchema's transcode takes 3.72x longer than Video-MME's
(6.25s vs. 1.68s) despite the video being only 2.42x longer (180s vs.
74.3s) — because it's also taller (448×336 vs. 448×252), and CPU-bound
`ffmpeg` encodes every pixel. Total pixels (frames × area) works out to
3.23x, much closer to the observed gap than duration alone.

**Full stage breakdown (`run_latency_deep_dive.py`, one extra level
below the table above)** — frames split into `fetch_video` (decord
decode+resize) vs. `image_processor` (PIL frames → tensors); codec split
into `cv_preinfer` (the CLI subprocess) vs. its own `image_processor`
(canvas JPEGs → tensors) vs. `other` (padding drop + position calc +
tokenize, not split further):

| Sample | Backend | Transcode | fetch_video | cv_preinfer | image_processor | other | VLM | Total |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| egoschema (180s) | frames | 0.00s | 0.22s | – | 0.10s | – | 3.74s | 4.06s |
| egoschema (180s) | codec | 6.15s | – | 0.49s | 0.06s | 0.59s | 4.06s | 11.36s |
| videomme (74s) | frames | 0.00s | 0.18s | – | 0.07s | – | 3.11s | 3.36s |
| videomme (74s) | codec | 1.66s | – | 0.38s | 0.05s | 0.53s | 4.58s | 7.21s |

![latency breakdown](latency_breakdown.png)

Three things fall out of this:
- **`fetch_video` (decord decode) is cheap and barely duration-dependent**
  (0.22s @180s vs. 0.18s @74s) — decord seeks directly to the 64 sampled
  frame indices rather than decoding the whole video, so raw decode was
  never actually the bottleneck for `frames`.
- **Codec's own `image_processor` (canvas → tensors) is tiny** (0.05–0.06s),
  even cheaper than frames' equivalent step — canvas packing itself
  isn't the problem either.
- **The real non-transcode, non-VLM bottleneck is codec's `other` bucket**
  (0.53–0.59s, comparable in size to `cv_preinfer`), and it's just as
  flat across the 2.4x duration difference as everything else in the
  codec path. This bundles `drop_padding_canvases` + position computation
  + tokenizing the rewritten (timestamp-injected) prompt — not split
  further here, so which of those three actually dominates is still
  open.

Bottom line: **transcode dominates codec's overhead by far** (6.15s/1.66s,
larger than every other codec-side stage combined); past that, no single
remaining stage explains the rest — `cv_preinfer` and `other` are both
mid-sized and both duration-independent, and codec's VLM pass itself
also runs consistently slower than frames' even at similar token counts
(see the token-matched comparison above).

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
