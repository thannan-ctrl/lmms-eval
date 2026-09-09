# Single-video smoke test

Quick sanity check for `llava_onevision2` on one EgoSchema video and one
Video-MME video, without going through the registered `lmms_eval` tasks
(their expected dataset layout/HF cache doesn't match the local mirror).

- Script: `run_single_video.py`
- Launcher (salloc/srun): `run_single_video.sh`

## Data source

Reads directly from local JSON + video files:

- `<DATA_ROOT>/egoschema/subset.json` + `egoschema/videos/<q_uid>.mp4`
- `<DATA_ROOT>/video_mme/questions.json` + `video_mme/videos/<video_id>.mp4`

Default `DATA_ROOT`: `/home/thannan/scratch/AutoGaze/data`.

Picks the first entry in each JSON whose video file exists locally.

## How it works

- Instantiates the real `Llava_OneVision2` model wrapper (frames backend,
  `messages_format=timestamp`) and reuses its actual
  `_build_messages` / chat-template / `generate` path.
- Prompt asks for brief grounded reasoning before a final answer letter
  (`Answer: X`), which is parsed and compared against local ground truth.

## Usage

```bash
bash examples/llava_onevision2_repro/run_single_video.sh
```

Env overrides: `DATA_ROOT`, `MODEL`, `NF` (num frames, default 32),
`PARTITION` (default `gb200nvl72_preprod`), `TIME` (default `08:00:00`).

Or run directly (e.g. inside an existing `salloc`/`srun --pty` shell):

```bash
python3 examples/llava_onevision2_repro/run_single_video.py \
    --data-root /home/thannan/scratch/AutoGaze/data \
    --num-frames 32
```

## Running in the README's Docker environment

To reproduce the exact environment from the top-level `README.md` § 1
(`dockerfile/Dockerfile`, `transformers==5.7.0`, `qwen-vl-utils==0.0.14`,
`codec-video-prep-legacy-exact==0.2.5.post2`) rather than an ad-hoc conda
env, run inside a GPU allocation:

```bash
srun --partition=A100x2 --nodes=1 --gres=gpu:1 --time=08:00:00 \
  bash -c '
    cd /home/scratch.thannan_wwfo/LLaVA-OneVision-2/lmms-eval
    docker build -t lmms-eval-ov2:latest -f dockerfile/Dockerfile .

    mkdir -p /home/thannan/scratch/hf_cache
    docker run --rm --gpus all --ipc=host --shm-size=16g --network=host \
      --user $(id -u):$(id -g) -e HOME=/tmp \
      -v $(pwd):/workspace/lmms-eval \
      -v /home/thannan/scratch/hf_cache:/hf_cache \
      -v /home/thannan/scratch/AutoGaze/data:/data \
      -e HF_HOME=/hf_cache \
      lmms-eval-ov2:latest bash -c "
        cd /workspace/lmms-eval
        pip install --user -e . --no-deps
        python3 -m pip install --user \
            --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ \
            codec-video-prep-legacy-exact==0.2.5.post2 || true
        python3 examples/llava_onevision2_repro/run_single_video.py \
            --data-root /data --num-frames 32
      "
  '
```

Two gotchas found while getting this working, both cluster-specific (not
README bugs):

- **Partition/arch**: `gb200nvl72_preprod` nodes are `aarch64`
  (Grace CPU); the Dockerfile's `pip install ... decord ...` fails there
  since `decord` has no aarch64 PyPI wheels. Use an `x86_64` A100
  partition instead (`A100x2` worked; matches the README's "verified on
  8 × A100-80GB" environment). Also note `/home/scratch.thannan_wwfo`
  (this repo + `AutoGaze/data`) is only mounted on some partitions —
  confirm with `ls` before assuming a partition can reach it.
- **NFS root-squash**: running the container as root (`docker run`'s
  default) gets `PermissionError` writing into the NFS-mounted repo
  (`pip install -e .` egg-info) and HF cache dirs. Fix: add
  `--user $(id -u):$(id -g) -e HOME=/tmp` so the container runs as the
  host user instead of root.

## Status

Executed successfully on `A100x2` (x86_64) via the Docker path above,
2026-09-09:

| Sample | Predicted | Ground truth |
|---|:-:|:-:|
| `egoschema/0074f737-11cb-497d-8d07-77c3a8127391` | D | D ✅ |
| `videomme/fFjv93ACGo8/001-1` | A | C ❌ |

Both requests completed without error, producing a parsed `Answer: X`
with grounded reasoning — confirms the environment + model + pipeline
work end-to-end. (One right, one wrong on n=1 each isn't a meaningful
accuracy signal; this is a pipeline smoke test, not a benchmark run.)
