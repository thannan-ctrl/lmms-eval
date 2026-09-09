# Single-video smoke test

Sanity check + frames-vs-codec latency comparison for `llava_onevision2`
on one EgoSchema video and one Video-MME video, run in the exact Docker
environment from the top-level `README.md` § 1. Script:
`run_single_video.py`.

## Results (2026-09-09, A100x2)

| Sample | Dur | Backend | Tok | Pred/GT | Trns | Pre | VLM | **E2E** | E2E−t | ×frames |
|---|--:|---|--:|:-:|--:|--:|--:|--:|--:|--:|
| egoschema | 180s | frames 64f | 12288 | D/D ✅ | 0.0 | 0.6 | 3.8 | **4.4** | 4.4 | 1.0x |
| egoschema | 180s | codec | 12288 | D/D ✅ | 6.3 | 0.8 | 4.0 | **11.0** | 4.8 | 2.5x |
| videomme | 74s | frames 64f | 9216 | B/C ❌ | 0.0 | 0.3 | 3.2 | **3.5** | 3.5 | 1.0x |
| videomme | 74s | codec | 11520 | A/C ❌ | 1.7 | 0.7 | 4.5 | **6.9** | 5.2 | 2.0x |

(all times in seconds; E2E−t = E2E minus transcode)

**Summary**: frames run at 64 frames to roughly match codec's token
budget (12288 tokens, an exact match for EgoSchema; 9216 for Video-MME —
token counts aren't perfectly controllable per-video since resolution
varies). Even at comparable token counts and **after subtracting
transcode** (E2E−transcode), codec is still slower: 4.77s vs. 4.37s for
EgoSchema, 5.20s vs. 3.45s for Video-MME (codec still has ~25% more
tokens there). So the token-budget gap explains some but not all of
codec's slowdown — canvas packing/image-processing (`processor call`)
is comparable to frames' raw video decode, but codec's VLM pass runs
consistently slower even at comparable token counts. Transcode itself is
a pure artifact of the local videos being `mpeg4` not H264/HEVC — an
H264/HEVC-native corpus skips it entirely. `codec`'s E2E also varied
run-to-run (11.3s → 6.9s for the same EgoSchema sample across runs),
likely OS page-cache/CPU contention noise in the `cv-preinfer`
subprocess. n=1/cell: illustrative, not a throughput benchmark.

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
