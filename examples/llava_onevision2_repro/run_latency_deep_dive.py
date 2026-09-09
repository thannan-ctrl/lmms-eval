#!/usr/bin/env python3
"""Fine-grained latency breakdown for llava_onevision2, frames vs. codec,
on exactly the two SMOKE_TEST.md videos (EgoSchema + Video-MME).

Goes one level deeper than run_single_video.py's build_messages /
chat_template / processor split: separates the actual video-decode call
from everything else inside each backend's preprocessing path, so we can
see which stages scale with video duration/resolution and which don't.

Frames stages: fetch_video (decord decode + qwen_vl_utils resize) ->
  image_processor (PIL frames -> pixel_values) -> vlm.
Codec stages: cv-preinfer subprocess (readiness scoring + canvas
  selection/writing) -> image_processor (canvas JPEGs -> pixel_values,
  via the checkpoint's codec_image_processor_outputs) -> "other" (padding
  drop, position computation, tokenize -- kept as one small remainder
  bucket) -> vlm.

Writes a grouped-bar PNG to <DATA_ROOT>/latency_breakdown.png (ad hoc,
exploratory -- not part of the resumable dataset eval).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import torch  # noqa: E402

from run_single_video import (  # noqa: E402
    build_mcq_prompt,
    load_egoschema_sample,
    load_videomme_sample,
    transcode_to_h264,
    video_duration_seconds,
)

_timing: dict[str, float] = {}


def _timed(name: str, fn):
    def wrapper(*a, **kw):
        t0 = time.time()
        out = fn(*a, **kw)
        _timing[name] = _timing.get(name, 0.0) + (time.time() - t0)
        return out
    return wrapper


def instrument():
    # Frames: decord decode + qwen_vl_utils resize.
    import qwen_vl_utils

    qwen_vl_utils.fetch_video = _timed("fetch_video", qwen_vl_utils.fetch_video)
    # llava_onevision2._process_video_with_timestamp does
    # `from qwen_vl_utils import fetch_video` locally each call, so patching
    # the qwen_vl_utils module attribute (above) is what actually takes
    # effect -- a from-import binds a new local name at call time here.

    # Codec: cv-preinfer subprocess (already used in run_single_video.py).
    from lmms_eval.models.chat import llava_onevision2 as l2mod

    l2mod._process_codec_video_tuned = _timed(
        "cv_preinfer", l2mod._process_codec_video_tuned
    )

    # Codec: canvas image processing (checkpoint-bundled, dynamically
    # loaded). Resolve it once up front so it's cached in sys.modules,
    # patch that cached module's attribute -- later dynamic-module lookups
    # for the same module path reuse the sys.modules entry, so they'll see
    # our patched function too.
    return l2mod  # caller resolves model.pretrained -> module path after model load


def instrument_codec_image_processor(pretrained: str):
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    module_path = f"{pretrained}--codec_video_processing_llava_onevision2.codec_image_processor_outputs"
    fn = get_class_from_dynamic_module(module_path, pretrained)
    mod = sys.modules.get(fn.__module__)
    if mod is not None and hasattr(mod, "codec_image_processor_outputs"):
        mod.codec_image_processor_outputs = _timed(
            "codec_image_processor", mod.codec_image_processor_outputs
        )
        return True
    return False


def run_frames(model, sample: dict, task_hint: str) -> dict:
    from lmms_eval.protocol import ChatMessages

    model.video_backend = "frames"
    prompt = build_mcq_prompt(sample["question"], sample["options"])
    cm = ChatMessages(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "video", "url": sample["video_path"]},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
    )
    _timing.clear()
    hf_messages, pil_images, _urls, _subs = model._build_messages(cm, task=task_hint)
    text = model.processor.apply_chat_template([hf_messages], tokenize=False, add_generation_prompt=True)
    t0 = time.time()
    inputs = model.processor(text=text, images=pil_images, videos=None, return_tensors="pt", padding=True)
    torch.cuda.synchronize()
    image_processor_s = time.time() - t0

    inputs = {k: (v.to(model._device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
    gen_args = dict(inputs)
    gen_args.pop("mm_token_type_ids", None)
    gen_args.update(
        eos_token_id=model.tokenizer.eos_token_id,
        pad_token_id=model.tokenizer.pad_token_id or model.tokenizer.eos_token_id,
        max_new_tokens=256, num_beams=1, do_sample=False, use_cache=True,
    )
    t0 = time.time()
    with torch.inference_mode():
        model.model.generate(**gen_args)
    torch.cuda.synchronize()
    vlm_s = time.time() - t0

    return {
        "backend": "frames",
        "fetch_video_s": _timing.get("fetch_video", 0.0),
        "image_processor_s": image_processor_s,
        "vlm_s": vlm_s,
        "transcode_s": 0.0,
        "cv_preinfer_s": 0.0,
        "other_s": 0.0,
    }


def run_codec(model, sample: dict, task_hint: str, video_path: str, transcode_s: float) -> dict:
    from lmms_eval.protocol import ChatMessages

    model.video_backend = "codec"
    model.codec_config = {"target_canvas": 64}
    prompt = build_mcq_prompt(sample["question"], sample["options"])
    cm = ChatMessages(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "video", "url": video_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
    )
    _timing.clear()
    hf_messages, _pil, video_urls, sub_dicts = model._build_messages(cm, task=task_hint)
    text = model.processor.apply_chat_template([hf_messages], tokenize=False, add_generation_prompt=True)

    t0 = time.time()
    inputs = model._codec_call_processor(texts=text, flat_videos=video_urls, subtitle_dicts=sub_dicts)
    torch.cuda.synchronize()
    processor_total_s = time.time() - t0

    cv_preinfer_s = _timing.get("cv_preinfer", 0.0)
    image_processor_s = _timing.get("codec_image_processor", 0.0)
    other_s = max(0.0, processor_total_s - cv_preinfer_s - image_processor_s)

    inputs = {k: (v.to(model._device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
    gen_args = dict(inputs)
    gen_args.pop("mm_token_type_ids", None)
    gen_args.update(
        eos_token_id=model.tokenizer.eos_token_id,
        pad_token_id=model.tokenizer.pad_token_id or model.tokenizer.eos_token_id,
        max_new_tokens=256, num_beams=1, do_sample=False, use_cache=True,
    )
    t0 = time.time()
    with torch.inference_mode():
        model.model.generate(**gen_args)
    torch.cuda.synchronize()
    vlm_s = time.time() - t0

    return {
        "backend": "codec",
        "fetch_video_s": 0.0,
        "cv_preinfer_s": cv_preinfer_s,
        "image_processor_s": image_processor_s,
        "other_s": other_s,
        "vlm_s": vlm_s,
        "transcode_s": transcode_s,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/data")
    ap.add_argument("--model", default="lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct")
    ap.add_argument("--num-frames", type=int, default=64)
    ap.add_argument("--codec-target-canvas", type=int, default=64)
    ap.add_argument("--min-pixels", type=int, default=100352)
    ap.add_argument("--max-pixels", type=int, default=313600)
    ap.add_argument("--transcode-dir", default="/tmp/h264_transcoded")
    ap.add_argument("--out-png", default=None)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    instrument()

    from lmms_eval.models.chat.llava_onevision2 import Llava_OneVision2

    print(f"[deep-dive] loading {args.model} ...")
    model = Llava_OneVision2(
        pretrained=args.model, device="cuda", attn_implementation="flash_attention_2",
        min_pixels=args.min_pixels, max_pixels=args.max_pixels,
        max_num_frames=args.num_frames, fps=1.0, messages_format="timestamp",
        video_backend="frames", codec_target_canvas=args.codec_target_canvas,
    )
    patched = instrument_codec_image_processor(args.model)
    print(f"[deep-dive] codec_image_processor instrumented: {patched}")

    samples = {
        "egoschema": load_egoschema_sample(data_root),
        "videomme": load_videomme_sample(data_root),
    }

    rows = []
    for name, sample in samples.items():
        dur = sample["duration_s"]
        print(f"\n--- {name} (duration={dur:.1f}s) ---")

        r_frames = run_frames(model, sample, name)
        r_frames["name"] = name
        r_frames["duration_s"] = dur
        rows.append(r_frames)
        print(f"[frames] fetch_video={r_frames['fetch_video_s']:.2f}s "
              f"image_processor={r_frames['image_processor_s']:.2f}s vlm={r_frames['vlm_s']:.2f}s")

        transcode_dir = Path(args.transcode_dir)
        video_path, transcode_s = transcode_to_h264(sample["video_path"], transcode_dir)
        r_codec = run_codec(model, sample, name, video_path, transcode_s)
        r_codec["name"] = name
        r_codec["duration_s"] = dur
        rows.append(r_codec)
        print(f"[codec] transcode={r_codec['transcode_s']:.2f}s cv_preinfer={r_codec['cv_preinfer_s']:.2f}s "
              f"image_processor={r_codec['image_processor_s']:.2f}s other={r_codec['other_s']:.2f}s "
              f"vlm={r_codec['vlm_s']:.2f}s")

    print("\n=== full breakdown ===")
    header = (
        f"{'sample':<10}{'backend':<8}{'dur_s':>7}{'transcode':>11}{'fetch_video':>13}"
        f"{'cv_preinfer':>13}{'image_proc':>12}{'other':>8}{'vlm':>8}{'total':>8}"
    )
    print(header)
    for r in rows:
        total = (
            r["transcode_s"] + r["fetch_video_s"] + r["cv_preinfer_s"]
            + r["image_processor_s"] + r["other_s"] + r["vlm_s"]
        )
        print(
            f"{r['name']:<10}{r['backend']:<8}{r['duration_s']:>7.1f}{r['transcode_s']:>11.2f}"
            f"{r['fetch_video_s']:>13.2f}{r['cv_preinfer_s']:>13.2f}{r['image_processor_s']:>12.2f}"
            f"{r['other_s']:>8.2f}{r['vlm_s']:>8.2f}{total:>8.2f}"
        )

    # Visualization: stacked bar per (sample, backend).
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        stages = ["transcode_s", "fetch_video_s", "cv_preinfer_s", "image_processor_s", "other_s", "vlm_s"]
        colors = ["#999999", "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
        labels = ["transcode", "fetch_video (decode)", "cv_preinfer", "image_processor", "other", "vlm"]

        fig, ax = plt.subplots(figsize=(8, 5))
        x = list(range(len(rows)))
        bottoms = [0.0] * len(rows)
        for stage, color, label in zip(stages, colors, labels):
            vals = [r[stage] for r in rows]
            ax.bar(x, vals, bottom=bottoms, color=color, label=label)
            bottoms = [b + v for b, v in zip(bottoms, vals)]
        ax.set_xticks(x)
        ax.set_xticklabels([f"{r['name']}\n{r['backend']}\n({r['duration_s']:.0f}s)" for r in rows])
        ax.set_ylabel("seconds")
        ax.set_title("llava_onevision2 latency breakdown: frames vs. codec")
        ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8)
        fig.tight_layout()
        out_png = Path(args.out_png) if args.out_png else data_root / "latency_breakdown.png"
        fig.savefig(out_png, dpi=150)
        print(f"\n[deep-dive] wrote {out_png}")
    except Exception as e:  # noqa: BLE001
        print(f"[deep-dive] plotting failed: {e}")


if __name__ == "__main__":
    main()
