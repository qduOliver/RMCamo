# RMCamo: Reference-guided Multimodal Camouflaged Image Generation

<p align="center">
  <img src="assets/fig4_rmcamo_overview.png" alt="RMCamo framework overview (Figure 4)" width="92%" />
</p>

**RMCamo** is an environment-aware camouflage generation framework in two stages:

1. **Environment-aware Control Net** — takes a *salient* RGB image and mask plus an environment cue (e.g. target color / style), and produces a **reference** RGB together with aligned **depth** and **mask**.
2. **RMCamo (Stage 2)** — a reference-guided multimodal module that fuses the reference image, **global / local text**, **depth & mask**, and an **[IP-Adapter](https://github.com/tencent-ailab/IP-Adapter)–style image prompt** branch: CLIP vision `image_embeds` → **`ImageProjModel`** → extra cross-attention tokens. Trainable blocks include **semantic–mask decoupling**, **pixel-level semantic alignment**, **object-level semantic-aware mask attention**, and **depth–mask coherence** guidance toward the UNet (full diffusion wiring comes with the future checkpoint release).

This repository releases **research code structure**, configuration, and a **Qwen-based text pipeline** for reproducibility. **Pretrained weights, training datasets, and quantitative results are not included yet** — we plan to release them in a future update after cleanup and legal review.

---

## Repository layout (similar spirit to IP-Adapter-style projects)

```
RMCamo/
├── assets/                    # Figures for README
├── configs/default.yaml       # Hyper-parameters skeleton
├── rmcamo/
│   ├── models/                # Stage 1 & 2 + IP-Adapter branch + prompt fusion
│   └── text/                  # Qwen caption + control-phrase utilities
├── scripts/
│   ├── extract_text_qwen.py   # CLI: image [+mask] → JSON caption bundle
│   └── smoke_test_forward.py  # Tensor shape check (requires torch)
├── requirements.txt
├── pyproject.toml
├── LICENSE
└── README.md
```

Install in editable mode:

```bash
cd RMCamo
pip install -e ".[full]"
# or: pip install -r requirements.txt
```

---

## Qwen: from image to text & control prompts

We recommend a **two-step linguistic workflow** aligned with the “Text + Control Prompt” block in the figure:

| Step | Model (example) | Role |
|------|-----------------|------|
| **A. Global caption** | `Qwen/Qwen2-VL-2B-Instruct` | One concise English sentence: subject–verb–object, concrete colors/textures; optional **second visual** = binary mask (encoded as grayscale RGB). |
| **B. Control phrases** | (Default) rule split on `build_control_prompts_from_caption` | Fast comma / “and” split into short phrases for conditioning. |
| **B′. (Optional)** | `Qwen/Qwen2.5-1.5B-Instruct` | Second LLM call asks for a **JSON array** of 4–8 short phrases (`refine_phrases_with_qwen_json`). |

**CLI example**

```bash
python scripts/extract_text_qwen.py \
  --image ./your_reference.png \
  --mask ./your_mask.png \
  --out ./caption_bundle.json \
  --refine
```

**Python API**

```python
from rmcamo.text import caption_with_qwen2_vl, refine_phrases_with_qwen_json

bundle = caption_with_qwen2_vl("ref.png", mask_path="mask.png")
bundle.phrases = refine_phrases_with_qwen_json(bundle.full_caption)
```

> **Note:** VLM inference needs sufficient GPU memory; follow the official Qwen2-VL and Qwen2.5 model cards for `transformers` version and `qwen-vl-utils` if you process multiple resolutions.

---

## Model code (skeleton)

- `EnvironmentAwareControlNet` — Stage 1 placeholder; replace forward with your environment-conditioned generator.
- `RMCamoStage2` — frozen-encoder I/O + trainable blocks + optional **`IPAdapterImageBranch`** (`ImageProjModel` matching IP-Adapter).
- **`CLIPImageEncoder`** + **`preprocess_reference_images`** — frozen CLIP vision encoding for reference PIL/tensors.
- **`load_image_proj_weights`** — load only `image_proj.*` from official `ip-adapter_sd15.safetensors` / checkpoint dict (UNet `ip_adapter.*` processors are **not** loaded here; use [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter) or Diffusers integration to swap attentions).
- **`concat_cross_attention_prompts`** — concatenate `[text_hidden, image_ip_tokens, extra_rmcamo_tokens]` for cross-attention (SD1.5 pattern).

**Example (image prompt tokens + text)**

```python
from rmcamo.models import (
    RMCamoStage2,
    CLIPImageEncoder,
    preprocess_reference_images,
    load_image_proj_weights,
    concat_cross_attention_prompts,
)
from transformers import CLIPImageProcessor

clip_path = "openai/clip-vit-large-patch14-336"  # match your IP-Adapter setup
proc = CLIPImageProcessor.from_pretrained(clip_path)
enc_clip = CLIPImageEncoder(clip_path).cuda()

stage2 = RMCamoStage2(enable_ip_adapter=True, clip_projection_dim=enc_clip.projection_dim).cuda()
load_image_proj_weights(stage2.ip_adapter_branch, "ip-adapter_sd15.safetensors")

# pixel_values = preprocess_reference_images(pil_image, proc)
# clip_embeds = enc_clip(pixel_values.cuda())
# _, _, ip_tok = stage2.forward_trainable_stack(enc, depth, mask, clip_image_embeds=clip_embeds)
# prompt = concat_cross_attention_prompts(encoder_hidden_states, ip_tok, extra_tokens)
```

Smoke test (requires `torch`):

```bash
python scripts/smoke_test_forward.py
```

---

## Roadmap / what is *not* in this drop

- End-to-end training & inference notebooks wired to **Stable Diffusion 1.5 UNet + IP-Adapter attention processors + ControlNet** (use upstream IP-Adapter repo to install processors, then feed `prompt` tensors from this repo).  
- Datasets and benchmark splits  
- Pretrained RMCamo weights and eval logs  

These items will be announced in **Releases** once ready.

---

## Citation

If you use this code or idea, please cite the paper when available (placeholder):

```bibtex
@article{rmcamo2026,
  title   = {Environment-aware Camouflage Generation with RMCamo},
  author  = {TBD},
  journal = {TBD},
  year    = {2026}
}
```

---

## Acknowledgements

Project structure is inspired by modular diffusion setups such as [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter). Vision-language captioning builds on the **Qwen** model family (Alibaba).

---

## License

MIT License — see [LICENSE](LICENSE).
