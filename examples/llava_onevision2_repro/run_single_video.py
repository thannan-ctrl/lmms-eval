#!/usr/bin/env python3
"""Smoke test: run LLaVA-OneVision-2-8B on a single EgoSchema video and a
single Video-MME video, straight from local JSON metadata (no HF `datasets`
dependency, no lmms-eval task/dataset plumbing).

Reads directly from the local mirrors already on disk:
  - <DATA_ROOT>/egoschema/{subset.json,videos/<q_uid>.mp4}
  - <DATA_ROOT>/video_mme/{questions.json,videos/<video_id>.mp4}

Usage (inside an salloc'd node / srun --pty shell):
    python examples/llava_onevision2_repro/run_single_video.py \
        --data-root /home/thannan/scratch/AutoGaze/data \
        --num-frames 32
"""

from __future__ import annotations

import argparse
import json
import re
import string
import subprocess
import time
from pathlib import Path

import cv2
import torch


def video_duration_seconds(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    try:
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        return frames / fps if fps else 0.0
    finally:
        cap.release()


def transcode_to_h264(src_path: str, out_dir: Path) -> tuple[str, float]:
    """cv-preinfer (the codec backend's canvas packer) only accepts
    H264/HEVC bitstreams. EgoSchema/Video-MME videos here are mpeg4, so
    transcode a copy first. Returns (output_path, transcode_latency_s).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (Path(src_path).stem + "_h264.mp4")
    if not out_path.exists():
        t0 = time.time()
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", src_path,
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-c:a", "copy",
                str(out_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        latency = time.time() - t0
    else:
        latency = 0.0  # already transcoded (cached across backend loop / reruns)
    return str(out_path), latency


def build_mcq_prompt(question: str, options: list[str]) -> str:
    """A more instructive MCQ prompt than a bare 'answer with the letter'.

    Asks for a short grounded justification before the final letter so a
    wrong answer is debuggable, but keeps the output format machine-parsable.
    """
    letters = string.ascii_uppercase
    options_block = "\n".join(f"{letters[i]}. {opt}" for i, opt in enumerate(options))
    return (
        "You are given a video and a multiple-choice question about it.\n\n"
        f"Question: {question}\n\n"
        f"Options:\n{options_block}\n\n"
        "Instructions:\n"
        "- Base your answer only on what is visibly shown in the video.\n"
        "- Consider the order and timing of events, not just isolated frames.\n"
        "- If two options seem close, pick the one best supported by direct "
        "visual evidence rather than assumption.\n\n"
        "Respond in exactly this format:\n"
        "Reasoning: <one or two sentences citing what you observed>\n"
        f"Answer: <a single letter, one of {', '.join(letters[: len(options)])}>"
    )


def load_egoschema_sample(data_root: Path):
    d = data_root / "egoschema"
    subset = json.loads((d / "subset.json").read_text())
    videos_dir = d / "videos"
    for item in subset:
        video_path = videos_dir / f"{item['q_uid']}.mp4"
        if video_path.exists():
            options = [item[f"option {i}"] for i in range(5)]
            gt = string.ascii_uppercase[item["answer"]] if "answer" in item else None
            return {
                "name": f"egoschema/{item['q_uid']}",
                "video_path": str(video_path),
                "duration_s": video_duration_seconds(str(video_path)),
                "question": item["question"],
                "options": options,
                "gt": gt,
            }
    raise RuntimeError("No egoschema subset.json entry has a matching local video file")


def load_videomme_sample(data_root: Path):
    d = data_root / "video_mme"
    questions = json.loads((d / "questions.json").read_text())
    videos_dir = d / "videos"
    for item in questions:
        video_path = videos_dir / f"{item['video_id']}.mp4"
        if video_path.exists():
            return {
                "name": f"videomme/{item['video_id']}/{item['question_id']}",
                "video_path": str(video_path),
                "duration_s": video_duration_seconds(str(video_path)),
                "question": item["question"],
                "options": item["options"],
                "gt": item.get("answer"),
            }
    raise RuntimeError("No video_mme questions.json entry has a matching local video file")


def run_one(model, sample: dict, task_hint: str, video_path: str | None = None, transcode_latency_s: float = 0.0) -> dict:
    """Run one sample through `model` (video_backend is whatever the caller
    already set on the model instance) and time the stages that matter for
    a backend comparison: transcode (codec only, 0 for frames), video
    preprocessing (frame extraction for "frames", canvas packing /
    cv-preinfer for "codec"), and the VLM forward pass (model.generate).
    """
    from lmms_eval.protocol import ChatMessages

    video_path = video_path or sample["video_path"]
    prompt = build_mcq_prompt(sample["question"], sample["options"])
    raw_messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "url": video_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    cm = ChatMessages(**{"messages": raw_messages})

    # Stage 1: _build_messages. For "frames" this does the actual video
    # decode + per-frame extraction/resize (qwen_vl_utils fetch_video); for
    # "codec" it's cheap (just collects video URLs, no decode yet).
    t0 = time.time()
    hf_messages, pil_images, video_urls, sub_dicts = model._build_messages(cm, task=task_hint)
    t1 = time.time()

    # Stage 2: chat-template rendering (string templating, no video work).
    text = model.processor.apply_chat_template(
        [hf_messages], tokenize=False, add_generation_prompt=True
    )
    t2 = time.time()

    # Stage 3: processor call. For "codec" this is where cv-preinfer canvas
    # packing + image processing of the canvases happens (the heavy part);
    # for "frames" it's just image-processing the already-extracted frames.
    if model.video_backend == "codec":
        # `text` is already a list[str] (batch of 1) from apply_chat_template
        # above -- don't wrap it again.
        inputs = model._codec_call_processor(
            texts=text, flat_videos=video_urls, subtitle_dicts=sub_dicts
        )
    else:
        inputs = model.processor(
            text=text,
            images=pil_images if pil_images else None,
            videos=None,
            return_tensors="pt",
            padding=True,
        )
    torch.cuda.synchronize()
    t3 = time.time()

    build_messages_latency_s = t1 - t0
    chat_template_latency_s = t2 - t1
    processor_latency_s = t3 - t2
    t_pre_start, t_pre_end = t0, t3

    inputs = {k: (v.to(model._device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
    gen_args = dict(inputs)
    gen_args.pop("mm_token_type_ids", None)
    gen_args.update(
        eos_token_id=model.tokenizer.eos_token_id,
        pad_token_id=model.tokenizer.pad_token_id or model.tokenizer.eos_token_id,
        max_new_tokens=256,
        num_beams=1,
        do_sample=False,
        use_cache=True,
    )
    t_vlm_start = time.time()
    with torch.inference_mode():
        out = model.model.generate(**gen_args)
    torch.cuda.synchronize()
    t_vlm_end = time.time()

    # Number of vision tokens actually consumed by the LLM: total patches
    # (sum of t*h*w over image_grid_thw rows) divided by spatial_merge_size^2
    # -- same quantity for both backends (frames: per-frame images; codec:
    # canvases), since both route through the same Qwen2VL-style grid format.
    num_video_tokens = None
    grid_thw = inputs.get("image_grid_thw")
    if grid_thw is not None:
        merge_size = int(getattr(model.processor, "spatial_merge_size", 2))
        num_video_tokens = int(grid_thw.prod(dim=-1).sum().item()) // (merge_size ** 2)

    out = out[:, inputs["input_ids"].shape[-1] :]
    response = model.tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()

    m = re.search(r"Answer:\s*([A-Z])", response)
    pred = m.group(1) if m else None
    preprocess_latency_s = t_pre_end - t_pre_start
    vlm_latency_s = t_vlm_end - t_vlm_start
    return {
        "prompt": prompt,
        "response": response,
        "pred": pred,
        "backend": model.video_backend,
        "transcode_latency_s": transcode_latency_s,
        "build_messages_latency_s": build_messages_latency_s,
        "chat_template_latency_s": chat_template_latency_s,
        "processor_latency_s": processor_latency_s,
        "preprocess_latency_s": preprocess_latency_s,
        "vlm_latency_s": vlm_latency_s,
        "e2e_latency_s": transcode_latency_s + preprocess_latency_s + vlm_latency_s,
        "num_video_tokens": num_video_tokens,
        **sample,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/home/thannan/scratch/AutoGaze/data")
    ap.add_argument(
        "--model", default="lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct"
    )
    ap.add_argument("--num-frames", type=int, default=32, help="max_num_frames (keep small for a quick smoke test)")
    ap.add_argument("--min-pixels", type=int, default=100352)
    ap.add_argument("--max-pixels", type=int, default=313600)
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument(
        "--backends",
        default="frames,codec",
        help="comma-separated: frames, codec, or frames,codec (default) to compare both",
    )
    ap.add_argument(
        "--codec-target-canvas",
        type=int,
        default=64,
        help="codec_target_canvas / max_num_frames for the codec backend (README default range: 64-128)",
    )
    ap.add_argument(
        "--transcode-dir",
        default="/tmp/h264_transcoded",
        help="where to write H264 transcodes for the codec backend (cv-preinfer requires H264/HEVC input)",
    )
    args = ap.parse_args()

    data_root = Path(args.data_root)
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]

    from lmms_eval.models.chat.llava_onevision2 import Llava_OneVision2

    print(f"[smoke-test] loading {args.model} ...")
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

    samples = {
        "egoschema": load_egoschema_sample(data_root),
        "videomme": load_videomme_sample(data_root),
    }

    transcode_dir = Path(args.transcode_dir)

    all_results = []
    for task_hint, sample in samples.items():
        for backend in backends:
            # Mutate the already-loaded model instead of reinstantiating,
            # so the 8B weights are loaded exactly once.
            model.video_backend = backend
            if backend == "codec":
                model.codec_config = {"target_canvas": args.codec_target_canvas}
                video_path, transcode_latency_s = transcode_to_h264(
                    sample["video_path"], transcode_dir
                )
                print(
                    f"\n[transcode] {sample['video_path']} -> {video_path} "
                    f"({transcode_latency_s:.2f}s)"
                )
            else:
                video_path, transcode_latency_s = sample["video_path"], 0.0

            print(f"\n=== {sample['name']} [{backend}] ===")
            print(f"video: {video_path}  (duration={sample['duration_s']:.1f}s)")
            print(f"question: {sample['question']}")
            result = run_one(
                model, sample, task_hint,
                video_path=video_path, transcode_latency_s=transcode_latency_s,
            )
            print(f"--- model response ---\n{result['response']}")
            print(f"predicted: {result['pred']}  ground_truth: {result['gt']}")
            print(
                f"transcode={result['transcode_latency_s']:.2f}s  "
                f"build_messages={result['build_messages_latency_s']:.2f}s  "
                f"chat_template={result['chat_template_latency_s']:.3f}s  "
                f"processor={result['processor_latency_s']:.2f}s  "
                f"(preprocess_total={result['preprocess_latency_s']:.2f}s)  "
                f"vlm={result['vlm_latency_s']:.2f}s  "
                f"e2e={result['e2e_latency_s']:.2f}s  "
                f"num_video_tokens={result['num_video_tokens']}"
            )
            all_results.append(result)

    if len(backends) > 1:
        print("\n=== latency comparison (frames = baseline) ===")
        header = (
            f"{'sample':<40}{'backend':<10}{'dur_s':>8}{'tokens':>8}{'transcode_s':>12}"
            f"{'preprocess_s':>14}{'vlm_s':>10}{'e2e_s':>10}{'e2e-transcode_s':>16}{'e2e_vs_frames':>16}"
        )
        print(header)
        by_sample: dict[str, dict[str, dict]] = {}
        for r in all_results:
            by_sample.setdefault(r["name"], {})[r["backend"]] = r
        for name, per_backend in by_sample.items():
            baseline_e2e = per_backend.get("frames", {}).get("e2e_latency_s")
            for backend, r in per_backend.items():
                delta = (
                    f"{r['e2e_latency_s'] / baseline_e2e:.2f}x"
                    if baseline_e2e
                    else "-"
                )
                e2e_minus_transcode = r["e2e_latency_s"] - r["transcode_latency_s"]
                print(
                    f"{name:<40}{backend:<10}{r['duration_s']:>8.1f}"
                    f"{r['num_video_tokens']:>8}{r['transcode_latency_s']:>12.2f}"
                    f"{r['preprocess_latency_s']:>14.2f}"
                    f"{r['vlm_latency_s']:>10.2f}{r['e2e_latency_s']:>10.2f}"
                    f"{e2e_minus_transcode:>16.2f}{delta:>16}"
                )


if __name__ == "__main__":
    main()
