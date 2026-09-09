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
from pathlib import Path

import torch


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
                "question": item["question"],
                "options": item["options"],
                "gt": item.get("answer"),
            }
    raise RuntimeError("No video_mme questions.json entry has a matching local video file")


def run_one(model, sample: dict, task_hint: str) -> dict:
    from lmms_eval.protocol import ChatMessages

    prompt = build_mcq_prompt(sample["question"], sample["options"])
    raw_messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "url": sample["video_path"]},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    cm = ChatMessages(**{"messages": raw_messages})
    hf_messages, pil_images, _video_urls, _sub_dicts = model._build_messages(cm, task=task_hint)

    text = model.processor.apply_chat_template(
        [hf_messages], tokenize=False, add_generation_prompt=True
    )
    inputs = model.processor(
        text=text,
        images=pil_images if pil_images else None,
        videos=None,
        return_tensors="pt",
        padding=True,
    )
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
    with torch.inference_mode():
        out = model.model.generate(**gen_args)
    out = out[:, inputs["input_ids"].shape[-1] :]
    response = model.tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()

    m = re.search(r"Answer:\s*([A-Z])", response)
    pred = m.group(1) if m else None
    return {"prompt": prompt, "response": response, "pred": pred, **sample}


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
    args = ap.parse_args()

    data_root = Path(args.data_root)

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
        video_backend="frames",
    )

    samples = {
        "egoschema": load_egoschema_sample(data_root),
        "videomme": load_videomme_sample(data_root),
    }

    for task_hint, sample in samples.items():
        print(f"\n=== {sample['name']} ===")
        print(f"video: {sample['video_path']}")
        print(f"question: {sample['question']}")
        result = run_one(model, sample, task_hint)
        print(f"--- model response ---\n{result['response']}")
        print(f"predicted: {result['pred']}  ground_truth: {result['gt']}")


if __name__ == "__main__":
    main()
