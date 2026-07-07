# NormalCrafter Clean — Architecture Notes

This document explains the runtime architecture of the commented implementation. The executable behavior is unchanged from the validated clean build; the added material is documentation only.

## 1. Design goal

The implementation is built around one ownership rule:

> Full-video intermediate tensors live on CPU. CUDA only receives the component and chunk/window currently being processed.

This prevents ComfyUI's graph cache from retaining a complete CUDA output and prevents total clip length from directly determining peak VRAM.

## 2. File map

- `nodes.py` — thin ComfyUI integration and UI declarations.
- `normalcrafter_clean/engine.py` — model loading, device lifecycle, and the four inference stages.
- `normalcrafter_clean/preprocess.py` — lazy frame resizing, temporal tail repetition, and spatial padding.
- `normalcrafter_clean/unet.py` — inference-only SVD UNet adaptation for one CLIP embedding per frame.
- `normalcrafter_clean/windows.py` — pure deterministic temporal window scheduler.
- `tests/test_windows.py` — unit tests for coverage and overlap behavior.

## 3. End-to-end flow

```text
ComfyUI IMAGE [F,H,W,C], CPU or GPU
        │
        ▼
FrameSource
  - RGB only
  - detached CPU FP32
  - aspect-preserving resize metadata
  - symmetric spatial padding metadata
        │
        ├──────────────────────────────────────────────┐
        │                                              │
        ▼                                              ▼
CLIP encoder                                      VAE encoder
chunked RGB frames                                chunked RGB frames
        │                                              │
        ▼                                              ▼
CPU CLIP embeddings                              CPU RGB latents
        │                                              │
        └──────────────────────┬───────────────────────┘
                               ▼
                         Temporal UNet
                    one overlapping window at a time
                               │
                               ▼
                       CPU normal latents
                               │
                               ▼
                         VAE decoder
                         chunked decode
                               │
                               ▼
                   CPU ComfyUI IMAGE [F,H,W,3]
```

## 4. Model components

`NormalCrafterModel` owns each heavy module exactly once:

- `image_encoder`: CLIP vision encoder from Stable Video Diffusion.
- `vae`: NormalCrafter's fine-tuned temporal VAE.
- `unet`: NormalCrafter's spatio-temporal UNet.
- `scheduler`: Euler scheduler used for the released one-step inference path.

There is no global pipeline variable and no second wrapper object containing duplicate strong references.

## 5. Device modes

### `staged`

Only the active heavy component is on the inference device:

1. CLIP on CUDA, VAE and UNet on CPU.
2. VAE on CUDA, CLIP and UNet on CPU.
3. UNet on CUDA, CLIP and VAE on CPU.
4. VAE on CUDA, CLIP and UNet on CPU.

This minimizes VRAM but pays for several model transfers.

### `resident`

All heavy components are moved to the selected device and remain there during inference. This avoids model transfer overhead and is faster when VRAM permits it.

## 6. Temporal behavior

The released defaults are:

- window size: 14 frames
- step size: 10 frames
- regular overlap: 4 frames

Long clips are processed by fixed-size windows. The last window is anchored to the end of the clip so no tail frame is omitted. If that anchor is close to the previous window, the final overlap may be larger than four frames.

The next window starts with the previous prediction in its overlapping prefix. After inference, old and new overlap predictions are linearly crossfaded.

Only the immediately previous GPU prediction window is retained. Temporal GPU state is therefore bounded by `window_size`, not total clip length.

## 7. Short clips

A clip shorter than one temporal window is logically extended by repeating its final frame. This happens lazily in `FrameSource.get_bchw`; no permanently padded video tensor is created.

Only real frames are decoded and returned.

## 8. Spatial behavior

Input is never enlarged. If the longest side exceeds `max_resolution`, the clip is downscaled while preserving aspect ratio.

The resized image is symmetrically padded to a multiple of 64 for the VAE/UNet hierarchy. Padding is removed after decoding. With `output_size="original"`, the decoded normal map is then resized back to the original input dimensions.

Because interpolation blends XYZ components, optional normal renormalization restores unit vector length before mapping values from `[-1,1]` to `[0,1]`.

## 9. One-step inference

The runtime path is deterministic:

1. Normal latents start at zero.
2. Existing overlap predictions seed the next window prefix.
3. RGB latents are concatenated channel-wise with the normal state.
4. The UNet evaluates one scheduler timestep.
5. Euler produces the predicted normal latent.

There is intentionally no seed input because the published runtime does not start from random noise.

## 10. Memory invariants

The implementation relies on these invariants:

- ComfyUI receives a CPU output tensor.
- Full CLIP embeddings are CPU tensors.
- Full RGB latent video is a CPU tensor.
- Full normal latent video is a CPU tensor.
- Only one encoder/decoder chunk or UNet window is transferred to CUDA at once.
- Output tensors are preallocated rather than grown through repeated `torch.cat`.
- Temporary local tensors are explicitly dereferenced after each stage iteration.
- Exceptions still execute model offloading through `finally`.

`torch.cuda.empty_cache()` can release allocator-reserved blocks only after live references disappear. It is cleanup support, not the mechanism that fixes ownership.

## 11. Performance knobs

- `offload_mode="resident"` is the largest speed gain when VRAM is sufficient.
- Larger CLIP/VAE chunk sizes reduce transfer and launch overhead but increase peak VRAM.
- A larger `step_size` creates less overlap and fewer UNet windows, but changes temporal behavior and must never exceed `window_size`.
- Lower `max_resolution` reduces both VAE and UNet cost significantly.

The safest optimization strategy is to alter one knob at a time while measuring output consistency, runtime, `torch.cuda.max_memory_allocated()`, and repeated-run stability.

## 12. Concurrency

One `NormalCrafterModel` instance has a re-entrant lock around inference and device changes. Two graph executions cannot simultaneously move the same modules between CPU and CUDA or mutate the shared scheduler.

Parallelism should use separate model instances only when the hardware has enough memory for them.
