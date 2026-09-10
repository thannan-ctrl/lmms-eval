# Frames vs. codec: does `llava_onevision2` run, and is codec worth it?

`llava_onevision2` can feed video to the model two ways: **`frames`**
(sample 64 frames evenly, feed them straight in) or **`codec`** (use a
separate tool, `cv-preinfer`, to pick the 64 "best" frames using signals
already sitting in the video's H264 encoding, instead of sampling
blindly). Ran both, on EgoSchema (500 videos) and Video-MME (1395
questions), in the exact Docker setup from `README.md` on `A100x2`.

## TL;DR

**Codec is 6-7x slower end-to-end for basically no accuracy gain.**

| Dataset | Backend | n | Accuracy | **E2E latency/sample** |
|---|---|--:|--:|--:|
| EgoSchema | frames | 500 | 69.4% | **3.81s** |
| EgoSchema | codec | 500 | 69.6% | **22.46s** |
| Video-MME | frames | 1395 | 62.2% | **3.26s** |
| Video-MME | codec | 1395 | 63.2% | **22.47s** |

Accuracy moves by ~0.2-1.0 points either way (noise-level for this n) —
codec buys almost nothing here. Full latency breakdown:

| Dataset | Backend | Tokens | Transcode | cv_preinfer | image_proc | other | **E2E** |
|---|---|--:|--:|--:|--:|--:|--:|
| egoschema | frames | 11760 | 0.00s | – | 0.09s | 0.02s | **3.81s** |
| egoschema | codec | 11925 | 6.26s | 11.05s | 0.06s | 0.54s | **22.46s** |
| videomme | frames | 8998 | 0.00s | – | 0.08s | 0.02s | **3.26s** |
| videomme | codec | 9955 | 7.53s | 10.32s | 0.05s | 0.55s | **22.47s** |

Two things drive nearly all of the gap:
1. **Transcode** — `cv-preinfer` only accepts H264/HEVC; our source
   videos are old-format `mpeg4`, so every codec run pays for an `ffmpeg`
   conversion first (skipped entirely if your video is already
   H264/HEVC).
2. **`cv_preinfer` itself** (the frame-selection step below) — this is
   the real cost, averaging 10-11s/video, way more than transcoding.

(ViT/LLM only got instrumented partway through this run, so it's just a
53-sample partial for Video-MME, not reliable at full scale: codec
`vit=0.03s llm=4.25s`, frames `vit=0.02s llm=3.42s`.)

## What "512→64 canvases" actually means

![how codec picks its canvases](codec_canvas_concept.png)

`cv-preinfer` (package `codec-video-prep-legacy-exact`) is a separate
tool this repo shells out to — it does the picking, not the model. One
call does:

1. Uniformly sample **512 candidate frames** from the video.
2. Score each one for "readiness" using **bit-cost and motion-vector
   data read straight from the H264 bitstream** — no full pixel decode
   needed to score a frame, which is also why the tool needs H264/HEVC
   input specifically.
3. Keep the best-scoring **64 frames** (4 out of every 32-frame group)
   and fully decode just those. Each kept frame becomes its own canvas
   image — one frame per canvas, not a multi-frame collage (confirmed by
   how the checkpoint code stamps one uniform timestamp per canvas).

**This is *not* the LLaVA-OneVision-2 companion paper's method.** It
reuses the same general idea — a codec's own compression decisions
already mark where a video's information-dense content is, so reuse
that instead of a separate analysis pass — but a different mechanism
entirely. Checked directly against the paper ([*OneVision-Encoder:
Codec-Aligned Sparsity*](https://arxiv.org/abs/2602.08683)) and its
official code
([github.com/EvolvingLMMs-Lab/OneVision-Encoder](https://github.com/EvolvingLMMs-Lab/OneVision-Encoder)):
the paper sparsifies **patches within frames** (every I-frame patch
kept, P-frame patches pruned to a fixed budget — no frame ever dropped
as a whole unit) using **HEVC**. `cv-preinfer` drops **whole frames**
(keeps 64 of 512, full patch grid on survivors) using **H264**, and
shares no code or terminology with the paper or its repo (`readiness`,
`bitcost`, `canvas`, `group_size` appear in neither).

Also: two separate, unrelated decodes happen across the two backends.
`ffmpeg` decodes the original `mpeg4` and re-encodes to H264 for
codec's `transcode` step, while `decord` separately decodes the
*original* `mpeg4` for frames' `fetch_video` step — neither backend's
decode touches the other's.

**Other quirks worth knowing:**
- `cv_preinfer`'s cost is **not duration-independent** — a single cherry-picked
  video made it look flat (~0.5s), but across the real datasets it
  averages 20-25x higher (10-11s), varying a lot sample to sample.
- The "512 candidates" figure barely shrinks with video length (only
  drops below ~17s videos) — so scanning candidates gets slower on
  longer videos even though the final canvas count stays 64.
- `--num-frames` does nothing for codec — that flag only controls the
  `frames` path. Codec's canvas count is set separately, via
  `--codec-target-canvas`.

(Single-video numbers below are illustrative, n=1 per cell — the table
above, from the full 500/1395-sample run, is the real comparison.)

<details>
<summary>Single-video micro-benchmark (2 videos, for sanity-checking the mechanism above)</summary>

| Sample | Dur | Backend | Frames | Tokens | Transcode | fetch_video | cv_preinfer | image_processor | other | VLM | **E2E** |
|---|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| egoschema | 180s | frames | 64 | 12288 | 0.00s | 0.22s | – | 0.10s | – | 3.74s | **4.06s** |
| egoschema | 180s | codec | 512→64 canvases | 12288 | 6.15s | – | 0.49s | 0.06s | 0.59s | 4.06s | **11.36s** |
| videomme | 74s | frames | 64 | 9216 | 0.00s | 0.18s | – | 0.07s | – | 3.11s | **3.36s** |
| videomme | 74s | codec | 512→64 canvases | 11520 | 1.66s | – | 0.38s | 0.05s | 0.53s | 4.58s | **7.21s** |

![latency breakdown](latency_breakdown.png)

Bigger/longer videos cost more to transcode, and not just
proportionally — ffmpeg touches every pixel, so a video 2.4x longer but
also a bit taller cost 3.7x more here.

</details>

## Why this can't run on GB200 (aarch64)

Not a model/GPU-support restriction — no GPU compatibility list is
published, and `transformers`/`torch`/CUDA all run fine on GB200. The
blocker is two auxiliary tools that only ship precompiled x86_64
binaries:

- **`decord`** (frames backend): no Linux aarch64 wheels on PyPI (or
  from the common substitute `eva-decord`). Its source **is** public
  ([github.com/dmlc/decord](https://github.com/dmlc/decord)), so a
  from-source aarch64 build is plausible in principle — not attempted
  here.
- **`codec-video-prep-legacy-exact`** (codec backend's `cv-preinfer`):
  installs fine on aarch64 but fails at runtime with `RuntimeError:
  cv_reader.read_video_cb not available` — its aarch64 wheel is a
  fallback that's missing the native backend, and **no source
  distribution is published anywhere** for this package. Genuine dead
  end without non-public source access.

So codec has no viable public path on GB200 today; frames might, with
effort. All numbers above are from `A100x2` (x86_64).

<details>
<summary><b>Reproduce</b></summary>

Data: `<DATA_ROOT>/egoschema/subset.json` + `.../video_mme/questions.json`
matched against local video files (default `DATA_ROOT`:
`/home/thannan/scratch/AutoGaze/data`).

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
        python3 examples/llava_onevision2_repro/run_dataset_eval.py \
            --data-root /data --backends frames,codec
      "
'
```

Key non-obvious flags:
- `A100x2`, not `gb200nvl72_preprod`: aarch64 partitions lack a working
  `decord`/`cv-preinfer` (see above).
- `--user $(id -u):$(id -g) -e HOME=/tmp`: NFS root-squash blocks
  `docker run`'s default root user from writing the mounted repo/cache.
- `-e LLAVA_CODEC_ONLINE_TUNED=1`: routes to the pinned
  `codec-video-prep-legacy-exact` CLI (the checkpoint's default path
  otherwise expects the README-forbidden `codec-video-prep` package).

`run_dataset_eval.py` checkpoints to a JSONL file
(`(dataset, sample_id, backend)` keyed) and resumes automatically —
safe to re-run after a SLURM timeout. Single-video variant:
`run_single_video.py --data-root /data --num-frames 64 --backends
frames,codec --codec-target-canvas 64`.

</details>
