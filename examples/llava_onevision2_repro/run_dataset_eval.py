#!/usr/bin/env python3
"""Whole-dataset frames-vs-codec latency + accuracy eval for llava_onevision2.

Extends run_single_video.py's per-sample instrumentation (transcode /
build_messages / chat_template / processor / cv-preinfer split / vlm /
e2e latency) to every EgoSchema subset video and every locally-available
Video-MME question, and adds an accuracy column.

Resumable across multiple SLURM allocations: each (dataset, sample_id,
backend) result is appended to a JSONL checkpoint file as it completes;
a rerun with the same --checkpoint path skips anything already done.

Usage (inside a Docker container per SMOKE_TEST.md's Reproduce section):
    python examples/llava_onevision2_repro/run_dataset_eval.py \
        --data-root /data --num-frames 64 --backends frames,codec \
        --checkpoint /data/dataset_eval_checkpoint.jsonl
"""

from __future__ import annotations

import argparse
import json
import string
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_single_video import (  # noqa: E402
    instrument_codec_image_processor,
    instrument_cv_preinfer,
    instrument_fine_grained,
    run_one,
    transcode_to_h264,
)


def load_egoschema_samples(data_root: Path, limit: int | None) -> list[dict]:
    d = data_root / "egoschema"
    subset = json.loads((d / "subset.json").read_text())
    videos_dir = d / "videos"
    out = []
    for item in subset:
        video_path = videos_dir / f"{item['q_uid']}.mp4"
        if not video_path.exists():
            continue
        options = [item[f"option {i}"] for i in range(5)]
        gt = string.ascii_uppercase[item["answer"]] if "answer" in item else None
        out.append(
            {
                "dataset": "egoschema",
                "sample_id": item["q_uid"],
                "name": f"egoschema/{item['q_uid']}",
                "video_path": str(video_path),
                "question": item["question"],
                "options": options,
                "gt": gt,
            }
        )
        if limit and len(out) >= limit:
            break
    return out


def load_videomme_samples(data_root: Path, limit: int | None) -> list[dict]:
    d = data_root / "video_mme"
    questions = json.loads((d / "questions.json").read_text())
    videos_dir = d / "videos"
    out = []
    for item in questions:
        video_path = videos_dir / f"{item['video_id']}.mp4"
        if not video_path.exists():
            continue
        out.append(
            {
                "dataset": "videomme",
                "sample_id": item["question_id"],
                "name": f"videomme/{item['video_id']}/{item['question_id']}",
                "video_path": str(video_path),
                "question": item["question"],
                "options": item["options"],
                "gt": item.get("answer"),
            }
        )
        if limit and len(out) >= limit:
            break
    return out


def load_checkpoint(path: Path) -> dict:
    done = {}
    if path.exists():
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                done[(r["dataset"], r["sample_id"], r["backend"])] = r
    return done


def append_checkpoint(path: Path, record: dict):
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def print_aggregate(done: dict):
    print("\n=== aggregate results ===")
    header = (
        f"{'dataset':<10}{'backend':<8}{'n':>5}{'acc':>8}{'transcode_s':>12}"
        f"{'fetch_video_s':>14}{'cv_preinfer_s':>14}{'image_proc_s':>13}{'other_s':>9}"
        f"{'vlm_s':>8}{'e2e_s':>8}"
    )
    print(header)
    by_key: dict[tuple, list] = {}
    for (dataset, _sample_id, backend), r in done.items():
        if "error" in r:
            continue
        by_key.setdefault((dataset, backend), []).append(r)
    for (dataset, backend), rs in sorted(by_key.items()):
        n = len(rs)
        correct = sum(1 for r in rs if r.get("pred") and r.get("gt") and r["pred"] == r["gt"])
        acc = correct / n if n else 0.0

        def avg(k):  # noqa: ANN001, ANN202
            vals = [r[k] for r in rs if r.get(k) is not None]
            return sum(vals) / len(vals) if vals else 0.0

        print(
            f"{dataset:<10}{backend:<8}{n:>5}{acc * 100:>7.1f}%{avg('transcode_latency_s'):>12.2f}"
            f"{avg('fetch_video_latency_s'):>14.2f}{avg('cv_preinfer_latency_s'):>14.2f}"
            f"{avg('image_processor_latency_s'):>13.2f}{avg('other_latency_s'):>9.2f}"
            f"{avg('vlm_latency_s'):>8.2f}{avg('e2e_latency_s'):>8.2f}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/home/thannan/scratch/AutoGaze/data")
    ap.add_argument("--model", default="lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct")
    ap.add_argument("--num-frames", type=int, default=64)
    ap.add_argument("--min-pixels", type=int, default=100352)
    ap.add_argument("--max-pixels", type=int, default=313600)
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--backends", default="frames,codec")
    ap.add_argument("--codec-target-canvas", type=int, default=64)
    ap.add_argument("--transcode-dir", default="/tmp/h264_transcoded")
    ap.add_argument("--datasets", default="egoschema,videomme")
    ap.add_argument("--limit", type=int, default=None, help="cap samples per dataset")
    ap.add_argument(
        "--checkpoint",
        default="/home/thannan/scratch/AutoGaze/dataset_eval_checkpoint.jsonl",
        help="JSONL results file; reruns with the same path resume, skipping completed work",
    )
    args = ap.parse_args()

    data_root = Path(args.data_root)
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    dataset_names = [d.strip() for d in args.datasets.split(",") if d.strip()]
    checkpoint_path = Path(args.checkpoint)
    transcode_dir = Path(args.transcode_dir)

    done = load_checkpoint(checkpoint_path)
    print(f"[dataset-eval] {len(done)} (dataset,sample,backend) results already in {checkpoint_path}")

    samples_by_dataset = {}
    if "egoschema" in dataset_names:
        samples_by_dataset["egoschema"] = load_egoschema_samples(data_root, args.limit)
    if "videomme" in dataset_names:
        samples_by_dataset["videomme"] = load_videomme_samples(data_root, args.limit)
    for name, samples in samples_by_dataset.items():
        print(f"[dataset-eval] {name}: {len(samples)} samples with a local video")

    total = sum(len(v) for v in samples_by_dataset.values()) * len(backends)
    remaining = total - len(done)
    print(f"[dataset-eval] {total} total (sample,backend) units; ~{remaining} remaining")

    from lmms_eval.models.chat.llava_onevision2 import Llava_OneVision2

    instrument_cv_preinfer()
    instrument_fine_grained()

    print(f"[dataset-eval] loading {args.model} ...")
    model = Llava_OneVision2(
        pretrained=args.model,
        device="cuda",
        attn_implementation="flash_attention_2",
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        max_num_frames=args.num_frames,
        fps=args.fps,
        messages_format="timestamp",
        video_backend=backends[0],
        codec_target_canvas=args.codec_target_canvas,
    )
    patched = instrument_codec_image_processor(args.model)
    print(f"[dataset-eval] codec_image_processor instrumented: {patched}")

    n_done_this_run = 0
    t_start = time.time()
    for dataset, samples in samples_by_dataset.items():
        for sample in samples:
            for backend in backends:
                key = (dataset, sample["sample_id"], backend)
                if key in done:
                    continue
                model.video_backend = backend
                if backend == "codec":
                    model.codec_config = {"target_canvas": args.codec_target_canvas}
                    video_path, transcode_latency_s = transcode_to_h264(
                        sample["video_path"], transcode_dir
                    )
                else:
                    video_path, transcode_latency_s = sample["video_path"], 0.0

                try:
                    result = run_one(
                        model, sample, dataset,
                        video_path=video_path, transcode_latency_s=transcode_latency_s,
                    )
                except Exception as e:  # noqa: BLE001
                    result = {
                        "dataset": dataset, "sample_id": sample["sample_id"],
                        "backend": backend, "error": str(e), "gt": sample.get("gt"),
                    }
                result.setdefault("dataset", dataset)
                result.setdefault("sample_id", sample["sample_id"])
                result.pop("prompt", None)  # keep checkpoint file small
                append_checkpoint(checkpoint_path, result)
                done[key] = result
                n_done_this_run += 1
                if n_done_this_run % 10 == 0:
                    elapsed = time.time() - t_start
                    print(
                        f"[dataset-eval] {n_done_this_run} done this run "
                        f"({len(done)}/{total} total), {elapsed / 60:.1f}min elapsed, "
                        f"~{elapsed / n_done_this_run:.1f}s/sample"
                    )

    print_aggregate(done)


if __name__ == "__main__":
    main()
