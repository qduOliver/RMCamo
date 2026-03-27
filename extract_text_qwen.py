#!/usr/bin/env python3
"""
CLI: image (+ optional mask) → caption + control phrases via Qwen2-VL (+ optional Qwen2.5 JSON refine).

Example:
  python scripts/extract_text_qwen.py \\
    --image path/to/ref.png \\
    --mask path/to/mask.png \\
    --out out/caption.json \\
    --refine

Environment:
  HF_HOME / TRANSFORMERS off-line caches as usual; GPU recommended.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rmcamo.text.qwen_pipeline import (
    caption_with_qwen2_vl,
    refine_phrases_with_qwen_json,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=str, required=True)
    p.add_argument("--mask", type=str, default="")
    p.add_argument("--out", type=str, default="caption_bundle.json")
    p.add_argument("--vl-model", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
    p.add_argument("--refine", action="store_true", help="Second pass with Qwen2.5 text model")
    p.add_argument("--text-model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    args = p.parse_args()

    mask_path = args.mask.strip() or None
    bundle = caption_with_qwen2_vl(
        args.image,
        mask_path=mask_path,
        model_id=args.vl_model,
    )
    if args.refine:
        bundle.phrases = refine_phrases_with_qwen_json(bundle.full_caption, text_model_id=args.text_model)
        bundle.raw["refined"] = True

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "full_caption": bundle.full_caption,
        "phrases": bundle.phrases,
        "raw": bundle.raw,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
