# Single-video smoke test: frames vs. codec

Quick check: does `llava_onevision2` actually run end-to-end, and how
much slower is the "codec" way of feeding it video vs. the plain
"frames" way? Ran on one EgoSchema video (3 min) and one Video-MME video
(74s), each two ways — `frames` (sample 64 frames, feed them straight
in) and `codec` (pack frames into composite "canvas" images first, via
a separate tool called `cv-preinfer`), in the exact Docker setup from
`README.md`.

## Results (2026-09-09, A100x2)

| Sample | Dur | Backend | Frames | Tokens | Transcode | fetch_video | cv_preinfer | image_processor | other | VLM | **E2E** | E2E−transcode | vs. frames | vs. frames, no transcode |
|---|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| egoschema | 180s | frames | 64 | 12288 | 0.00s | 0.22s | – | 0.10s | – | 3.74s | **4.06s** | 4.06s | 1.00x | 1.00x |
| egoschema | 180s | codec | 512→64 canvases | 12288 | 6.15s | – | 0.49s | 0.06s | 0.59s | 4.06s | **11.36s** | 5.21s | 2.80x | 1.28x |
| videomme | 74s | frames | 64 | 9216 | 0.00s | 0.18s | – | 0.07s | – | 3.11s | **3.36s** | 3.36s | 1.00x | 1.00x |
| videomme | 74s | codec | 512→64 canvases | 11520 | 1.66s | – | 0.38s | 0.05s | 0.53s | 4.58s | **7.21s** | 5.55s | 2.15x | 1.65x |

![latency breakdown](latency_breakdown.png)

### What "512→64 canvases" actually means

![how codec picks its canvases](codec_canvas_concept.png)

`cv_preinfer` is a separate, pip-installed tool (`codec-video-prep-legacy-exact`)
that codec shells out to for video prep — this repo just calls it and
hands it a video, it doesn't pick frames itself. In one call it:

1. Uniformly samples **512 candidate frames** from the video.
2. Scores each one for "readiness" using **bit-cost and motion-vector
   data read straight from the H264 bitstream** — no full pixel decode
   needed for scoring, which is why it only accepts H264/HEVC input
   (per the companion paper, [*OneVision-Encoder: Codec-Aligned
   Sparsity*](https://arxiv.org/abs/2602.08683) — the idea is that a
   codec's own compression decisions already mark where the
   information-dense parts of a video are, so reuse that instead of
   redoing the analysis from scratch).
3. Keeps the best-scoring **64 frames** (4 out of every 32-frame group),
   fully decodes just those, and packs them into canvas images.

Two separate video decodes happen in this pipeline, in two different
tools that share no work: `ffmpeg` decodes the original `mpeg4` and
re-encodes it to H264 (the `transcode` stage), while `decord` decodes
the *original* `mpeg4` directly for the `frames` backend's `fetch_video`
stage — codec's path never touches what `decord` does, and vice versa.

## Bottom line: codec is a lot slower, mostly one extra step

- **Biggest cost: converting the video format.** Our test videos are an
  old format (`mpeg4`); `cv-preinfer` only accepts newer ones
  (H264/HEVC), so we convert first — that alone eats 23-54% of codec's
  total time. Videos already in H264/HEVC skip this entirely.
- **Even without that conversion, codec is still 30-65% slower** at a
  similar amount of info fed to the model — the model itself just takes
  longer per token when its input comes from codec.
- **Bigger/longer videos take longer to convert**, and not just
  proportionally to length — ffmpeg touches every pixel, so a video
  2.4x longer but also a bit taller costs 3.7x more to convert.
- **Sampling frames barely depends on video length** — grabbing 64
  frames takes about the same time from a 74s video or a 3-minute one,
  since we jump straight to those frames instead of decoding everything.
- **codec secretly scans way more of the video than "64" suggests.**
  Under the hood it first looks at 512 candidate frames spread across
  the video before packing anything into the 64 canvases. That 512
  doesn't shrink with video length (unless the video's under ~17s) —
  but *fetching* those 512 frames still takes longer on a longer video.
- **`--num-frames` does nothing for codec** — it only controls the
  plain-frames path. Codec's size is set by a different flag,
  `--codec-target-canvas`.

(n=1 per cell — illustrative, not a real benchmark.)

## Whole-dataset eval (in progress, last updated 2026-09-10)

Same frames-vs-codec comparison, but run on every EgoSchema subset video
(500) and every locally-available Video-MME question (1395), with
accuracy this time. Script: `run_dataset_eval.py`, resumable via a JSONL
checkpoint (see Reproduce below). Codec uses the same sampling method
(`uniform_count`) and config (unscaled, identical for every video) as
the single-video test above. **EgoSchema is done; Video-MME is ~85%
through as of this snapshot — numbers below will keep shifting until it
finishes.**

| Dataset | Backend | n | Acc | E2E | Transcode | cv_preinfer | VLM |
|---|---|--:|--:|--:|--:|--:|--:|
| egoschema | frames | 500 | 69.4% | 3.81s | 0.00s | – | 3.47s |
| egoschema | codec | 500 | 69.6% | 22.46s | 6.26s | 11.05s | 4.56s |
| videomme | frames | 1180 | 65.4% | 3.18s | 0.00s | – | 2.87s |
| videomme | codec | 1179 | 65.4% | 18.23s | 5.55s | 8.08s | 4.00s |

At real scale, accuracy is a dead heat both datasets (69.4/69.6% on
EgoSchema, 65.4/65.4% so far on Video-MME) — codec's extra latency buys
nothing here. `cv_preinfer` also confirms what the single-video test
couldn't: it's *not* duration-independent in general — its average kept
climbing as more (and longer/heavier) videos were sampled (0.49s on the
one cherry-picked video → 11.05s average over all 500 EgoSchema videos).

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
            --data-root /data --num-frames 64 \
            --backends frames,codec --codec-target-canvas 64
      "
'
```

**Notes** (all cluster/data-specific, not README bugs):
- `A100x2`, not `gb200nvl72_preprod`: that partition is `aarch64` and
  `decord` has no aarch64 wheels. `A100x2` matches the README's
  "verified on 8 × A100-80GB" and has `/home/scratch.thannan_wwfo`
  mounted (confirm with `ls` — not all partitions/nodes do).
- `--user $(id -u):$(id -g) -e HOME=/tmp`: NFS root-squash blocks
  `docker run`'s default root user from writing the mounted repo/cache.
- `-e LLAVA_CODEC_ONLINE_TUNED=1`: the checkpoint's default codec path
  calls a `cv-preinfer` binary only the README-forbidden regular
  `codec-video-prep` package provides; this repo's "online tuned" path
  correctly calls the pinned `codec-video-prep-legacy-exact` CLI, opt-in
  via this env var.
- Transcode: `cv-preinfer` only accepts H264/HEVC; local videos are
  `mpeg4`, so the scripts transcode a throwaway H264 copy
  (`ffmpeg -c:v libx264 -preset fast -crf 23`) per video, timed
  separately as `transcode_latency_s`.

</details>
