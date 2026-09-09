#!/usr/bin/env python3
"""Ablation: how does cv-preinfer's candidate-frame pool size
(`num_sampled_frames`) affect codec-backend preprocessing latency?

`num_sampled_frames = (target_canvas // images_per_group) * group_size`
(clamped to the video's actual total frame count). Fixes
`target_canvas=64, images_per_group=4` and sweeps `group_size` so the
candidate pool goes from the default 512 up to each video's actual max
frame count (an intentionally oversized group_size that gets clamped by
the runtime to `min(num_sampled_frames(), total_frames)`).

Codec preprocessing ONLY (no VLM generate() call) -- transcode is done
once per video and reused across the sweep, since it doesn't depend on
group_size. Just the 2 SMOKE_TEST.md videos.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import cv2  # noqa: E402
import torch  # noqa: E402

from run_single_video import (  # noqa: E402
    build_mcq_prompt,
    instrument_codec_image_processor,
    instrument_cv_preinfer,
    load_egoschema_sample,
    load_videomme_sample,
    transcode_to_h264,
    _cv_preinfer_timing,
)


def total_frames(video_path: str) -> int:
    cap = cv2.VideoCapture(video_path)
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()


def run_codec_preprocess_once(model, sample: dict, task_hint: str, video_path: str) -> dict:
    from lmms_eval.protocol import ChatMessages

    model.video_backend = "codec"
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
    hf_messages, _pil, video_urls, sub_dicts = model._build_messages(cm, task=task_hint)
    text = model.processor.apply_chat_template([hf_messages], tokenize=False, add_generation_prompt=True)

    _cv_preinfer_timing.pop("last_s", None)
    t0 = time.time()
    model._codec_call_processor(texts=text, flat_videos=video_urls, subtitle_dicts=sub_dicts)
    torch.cuda.synchronize()
    total_s = time.time() - t0
    cv_preinfer_s = _cv_preinfer_timing.pop("last_s", 0.0)
    return {"total_s": total_s, "cv_preinfer_s": cv_preinfer_s, "rest_s": total_s - cv_preinfer_s}


def main():
    data_root = Path("/data")
    instrument_cv_preinfer()

    from lmms_eval.models.chat.llava_onevision2 import Llava_OneVision2

    model_name = "lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct"
    print(f"[ablation] loading {model_name} ...")
    model = Llava_OneVision2(
        pretrained=model_name, device="cuda", attn_implementation="flash_attention_2",
        min_pixels=100352, max_pixels=313600, max_num_frames=64, fps=1.0,
        messages_format="timestamp", video_backend="codec", codec_target_canvas=64,
    )
    instrument_codec_image_processor(model_name)

    samples = {
        "egoschema": load_egoschema_sample(data_root),
        "videomme": load_videomme_sample(data_root),
    }

    transcode_dir = Path("/tmp/h264_transcoded")
    images_per_group = 4
    target_canvas = 64
    # group_size values -> intended num_sampled_frames = (64//4)*group_size = 16*group_size
    group_sizes = [32, 64, 128, 256, 100000]  # last one is "oversized -> clamps to actual max"

    results = []
    for name, sample in samples.items():
        video_path, transcode_s = transcode_to_h264(sample["video_path"], transcode_dir)
        n_total = total_frames(video_path)
        print(f"\n--- {name}: total_frames={n_total}, transcode={transcode_s:.2f}s ---")

        for gs in group_sizes:
            model.codec_config = {
                "target_canvas": target_canvas,
                "group_size": gs,
                "images_per_group": images_per_group,
            }
            intended = (target_canvas // images_per_group) * gs
            actual = min(intended, n_total)
            r = run_codec_preprocess_once(model, sample, name, video_path)
            r.update(name=name, group_size=gs, intended=intended, actual=actual, n_total=n_total)
            results.append(r)
            print(
                f"  group_size={gs:>7}  num_sampled(intended->actual)={intended:>7}->{actual:<7}  "
                f"cv_preinfer={r['cv_preinfer_s']:.2f}s  rest={r['rest_s']:.2f}s  total={r['total_s']:.2f}s"
            )

    print("\n=== ablation summary ===")
    header = f"{'video':<10}{'num_sampled':>13}{'cv_preinfer_s':>15}{'rest_s':>9}{'total_s':>10}"
    print(header)
    for r in results:
        print(f"{r['name']:<10}{r['actual']:>13}{r['cv_preinfer_s']:>15.2f}{r['rest_s']:>9.2f}{r['total_s']:>10.2f}")

    # Line plot: latency vs. candidate pool size, one line per video.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        colors = {"egoschema": "#2a78d6", "videomme": "#eb6834"}
        fig, ax = plt.subplots(figsize=(6.5, 4.2), facecolor="#fcfcfb")
        ax.set_facecolor("#fcfcfb")
        for name in samples:
            rs = sorted([r for r in results if r["name"] == name], key=lambda r: r["actual"])
            xs = [r["actual"] for r in rs]
            ys = [r["cv_preinfer_s"] for r in rs]
            ax.plot(xs, ys, marker="o", color=colors[name], label=f"{name} (cv_preinfer)", linewidth=2)
        ax.set_xlabel("num_sampled_frames (candidate pool size)", fontsize=9)
        ax.set_ylabel("cv_preinfer latency (s)", fontsize=9)
        ax.set_title("codec candidate-pool size vs. cv-preinfer latency", fontsize=11, fontweight="bold", loc="left")
        ax.xaxis.grid(True, color="#e3e2dd", linewidth=1, zorder=0)
        ax.yaxis.grid(True, color="#e3e2dd", linewidth=1, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        out_png = data_root / "codec_sampling_ablation.png"
        fig.savefig(out_png, dpi=170, facecolor="#fcfcfb")
        print(f"\n[ablation] wrote {out_png}")
    except Exception as e:  # noqa: BLE001
        print(f"[ablation] plotting failed: {e}")


if __name__ == "__main__":
    main()
