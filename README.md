# ComfyUI NormalCrafter Clean

A ground-up ComfyUI inference implementation for **NormalCrafter**. It uses the released NormalCrafter UNet/VAE weights and the Stable Video Diffusion image encoder/scheduler, but does not reuse the research pipeline or the existing ComfyUI wrapper.

## Why this exists

The existing wrapper mixes node UI, downloads, global pipeline caching, device state, inference, retry logic, resizing, and cleanup. It can also return live CUDA tensors into ComfyUI's cache. This implementation makes ownership explicit:

- one model object owns each component exactly once;
- no module-level model globals;
- ComfyUI receives a CPU tensor;
- CLIP embeddings, RGB latents, and predicted normal latents are stored on CPU between stages;
- only the active chunk/window is moved to CUDA;
- the output latent video is preallocated instead of repeatedly grown with `torch.cat`;
- short clips are actually padded and then trimmed;
- cleanup runs through a single lifecycle path.

## Nodes

### NormalCrafter Clean - Load

Loads:

- `Yanrui95/NormalCrafter`: NormalCrafter UNet and normal VAE;
- `stabilityai/stable-video-diffusion-img2vid-xt`: CLIP vision encoder and Euler scheduler.

The model is initially kept on CPU. `auto` dtype selects FP16 on CUDA and FP32 on CPU.

### NormalCrafter Clean - Generate Normals

`staged` offload mode runs the model in four explicit phases:

1. CLIP frame embeddings;
2. RGB VAE encoding;
3. sliding-window UNet inference;
4. normal VAE decoding.

Only one heavy component is resident on CUDA at a time. `resident` keeps all components on the selected device and is faster, but needs substantially more VRAM.

The released model uses a deterministic one-step scheduler path. There is therefore intentionally no seed input.

### NormalCrafter Clean - Offload

Moves every model component to CPU and clears the CUDA allocator cache.

## Installation

Place the folder in `ComfyUI/custom_nodes/` and install dependencies inside the same Python environment as ComfyUI:

```bash
pip install -r requirements.txt
```

Restart ComfyUI afterward. The first loader execution downloads the Hugging Face weights unless `local_files_only` is enabled.

## Defaults

- window size: 14 frames
- step size: 10 frames
- temporal overlap: 4 frames in the regular case
- conditioning FPS: 7
- motion bucket: 127
- noise augmentation: 0
- output: original input resolution

The final normal vectors are renormalized after resizing by default. Disable this only for strict comparisons with older glue code.

## Validation status

The source is syntax-checked and the pure window scheduler has unit tests. It has **not** been executed here against the multi-gigabyte model weights on an NVIDIA GPU. The first real validation should compare a fixed clip against the research repository and record both visual output and `torch.cuda.memory_allocated()` across repeated runs.

## Attribution

NormalCrafter research code: MIT License. Hugging Face Diffusers-derived UNet execution structure: Apache License 2.0. See `NOTICE`.
