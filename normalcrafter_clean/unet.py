"""Inference-only NormalCrafter adaptation of Diffusers' temporal SVD UNet.

The released checkpoint uses the Stable Video Diffusion UNet architecture but
conditions each frame with its own CLIP embedding. Standard SVD commonly starts
from one image embedding and repeats it across the generated video. The override
below keeps Diffusers' block execution intact while accepting either convention.

Training-only machinery from the research project, including DINO supervision,
is deliberately absent: it shaped the learned weights but is not part of runtime
inference.
"""

# Adapted for inference from Hugging Face Diffusers (Apache-2.0) and
# Binyr/NormalCrafter (MIT). See NOTICE.
from __future__ import annotations

from typing import Union

import torch
from diffusers import UNetSpatioTemporalConditionModel

# Diffusers moved this output class between package paths. Supporting both keeps
# the node compatible across a broader range of ComfyUI environments without
# changing the actual UNet computation.
try:
    from diffusers.models.unets.unet_spatio_temporal_condition import (
        UNetSpatioTemporalConditionOutput,
    )
except ImportError:  # older Diffusers
    from diffusers.models.unet_spatio_temporal_condition import (  # type: ignore
        UNetSpatioTemporalConditionOutput,
    )


class NormalCrafterUNet(UNetSpatioTemporalConditionModel):
    """SVD UNet accepting one CLIP embedding per video frame.

    Input shape
    -----------
    ``sample`` is ``[batch, frames, channels, latent_h, latent_w]``. In the clean
    engine, channels are eight: four zero/previous normal-latent channels plus
    four encoded RGB-conditioning channels.

    Conditioning shape
    ------------------
    ``encoder_hidden_states`` may contain either:

    * one embedding per batch item, which is repeated over frames like SVD; or
    * one embedding per frame, already flattened as ``batch * frames``.

    Output shape mirrors the temporal input layout and contains the four predicted
    normal-latent channels expected by the Euler scheduler step.
    """

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        added_time_ids: torch.Tensor,
        return_dict: bool = True,
    ):
        """Run one spatio-temporal UNet evaluation.

        This follows Diffusers' native forward structure closely. The important
        NormalCrafter-specific behavior is the frame-wise conditioning validation
        and preservation rather than unconditional repetition of one embedding.
        """

        # Normalize scalar Python or zero-dimensional Torch timesteps into the
        # one-dimensional tensor form expected by the projection layer.
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            dtype = torch.float64 if isinstance(timestep, float) else torch.int64
            if sample.device.type == "mps":
                dtype = torch.float32 if isinstance(timestep, float) else torch.int32
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif timesteps.ndim == 0:
            timesteps = timesteps[None].to(sample.device)

        batch_size, num_frames = sample.shape[:2]

        # One scheduler timestep applies to every item in the inference batch.
        timesteps = timesteps.expand(batch_size)

        # Standard diffusion timestep embedding. Cast to sample dtype so half or
        # bfloat inference does not accidentally promote subsequent activations.
        t_emb = self.time_proj(timesteps).to(dtype=sample.dtype)
        emb = self.time_embedding(t_emb)

        # SVD's added conditioning carries FPS, motion bucket, and noise strength.
        # The projection output is added to the ordinary diffusion-time embedding.
        time_embeds = self.add_time_proj(added_time_ids.flatten())
        time_embeds = time_embeds.reshape(batch_size, -1).to(emb.dtype)
        emb = emb + self.add_embedding(time_embeds)

        # Diffusers' spatial blocks operate on batch*frames as the leading axis;
        # therefore every frame receives a copy of its video's time embedding.
        emb = emb.repeat_interleave(num_frames, dim=0)

        # [B, F, C, H, W] -> [B*F, C, H, W]. Temporal awareness is preserved by
        # image_only_indicator and temporal modules inside the SVD blocks.
        sample = sample.flatten(0, 1)

        # Accept standard SVD conditioning ([B, ...]) or NormalCrafter's per-frame
        # conditioning ([B*F, ...]). Reject every other count before entering deep
        # attention layers, where the resulting error would be far less readable.
        expected_frame_batch = batch_size * num_frames
        if encoder_hidden_states.shape[0] == batch_size:
            encoder_hidden_states = encoder_hidden_states.repeat_interleave(num_frames, dim=0)
        elif encoder_hidden_states.shape[0] != expected_frame_batch:
            raise ValueError(
                "encoder_hidden_states must contain either one embedding per batch "
                f"or one per frame; got {encoder_hidden_states.shape[0]} for "
                f"batch={batch_size}, frames={num_frames}"
            )

        # Initial projection maps the eight-channel latent input into the UNet's
        # first feature width defined by the downloaded checkpoint configuration.
        sample = self.conv_in(sample)

        # All frames are video frames, not isolated images. SVD temporal blocks use
        # this indicator to control image/video-specific blending behavior.
        image_only_indicator = torch.zeros(
            batch_size,
            num_frames,
            dtype=sample.dtype,
            device=sample.device,
        )

        # Encoder/down path. Residual feature maps are accumulated for the matching
        # decoder/up blocks, reproducing the ordinary UNet skip-connection topology.
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if getattr(downsample_block, "has_cross_attention", False):
                sample, residuals = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                )
            else:
                sample, residuals = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    image_only_indicator=image_only_indicator,
                )
            down_block_res_samples += residuals

        # Bottleneck with cross-attention and temporal processing at the lowest
        # spatial resolution, where receptive field is largest.
        sample = self.mid_block(
            hidden_states=sample,
            temb=emb,
            encoder_hidden_states=encoder_hidden_states,
            image_only_indicator=image_only_indicator,
        )

        # Decoder/up path. Each block consumes exactly as many stored residuals as
        # it has ResNet sub-blocks, then removes them from the residual stack.
        for upsample_block in self.up_blocks:
            residuals = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            if getattr(upsample_block, "has_cross_attention", False):
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=residuals,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=residuals,
                    image_only_indicator=image_only_indicator,
                )

        # Final normalization, activation, and convolution restore the checkpoint's
        # configured output channel count. Then recover explicit video dimensions.
        sample = self.conv_out(self.conv_act(self.conv_norm_out(sample)))
        sample = sample.reshape(batch_size, num_frames, *sample.shape[1:])

        # Match Diffusers' conventional tuple-or-dataclass API.
        if not return_dict:
            return (sample,)
        return UNetSpatioTemporalConditionOutput(sample=sample)
