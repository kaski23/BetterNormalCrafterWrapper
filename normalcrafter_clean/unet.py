# Adapted for inference from Hugging Face Diffusers (Apache-2.0) and
# Binyr/NormalCrafter (MIT). See NOTICE.
from __future__ import annotations

from typing import Union

import torch
from diffusers import UNetSpatioTemporalConditionModel

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

    Upstream SVD repeats one image embedding over the full video. NormalCrafter
    instead passes ``batch * frames`` embeddings. Everything else is ordinary
    inference-only UNet execution; training-only DINO and ControlNet branches are
    deliberately excluded.
    """

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        added_time_ids: torch.Tensor,
        return_dict: bool = True,
    ):
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            dtype = torch.float64 if isinstance(timestep, float) else torch.int64
            if sample.device.type == "mps":
                dtype = torch.float32 if isinstance(timestep, float) else torch.int32
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif timesteps.ndim == 0:
            timesteps = timesteps[None].to(sample.device)

        batch_size, num_frames = sample.shape[:2]
        timesteps = timesteps.expand(batch_size)

        t_emb = self.time_proj(timesteps).to(dtype=sample.dtype)
        emb = self.time_embedding(t_emb)

        time_embeds = self.add_time_proj(added_time_ids.flatten())
        time_embeds = time_embeds.reshape(batch_size, -1).to(emb.dtype)
        emb = emb + self.add_embedding(time_embeds)
        emb = emb.repeat_interleave(num_frames, dim=0)

        sample = sample.flatten(0, 1)

        expected_frame_batch = batch_size * num_frames
        if encoder_hidden_states.shape[0] == batch_size:
            encoder_hidden_states = encoder_hidden_states.repeat_interleave(num_frames, dim=0)
        elif encoder_hidden_states.shape[0] != expected_frame_batch:
            raise ValueError(
                "encoder_hidden_states must contain either one embedding per batch "
                f"or one per frame; got {encoder_hidden_states.shape[0]} for "
                f"batch={batch_size}, frames={num_frames}"
            )

        sample = self.conv_in(sample)
        image_only_indicator = torch.zeros(
            batch_size,
            num_frames,
            dtype=sample.dtype,
            device=sample.device,
        )

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

        sample = self.mid_block(
            hidden_states=sample,
            temb=emb,
            encoder_hidden_states=encoder_hidden_states,
            image_only_indicator=image_only_indicator,
        )

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

        sample = self.conv_out(self.conv_act(self.conv_norm_out(sample)))
        sample = sample.reshape(batch_size, num_frames, *sample.shape[1:])

        if not return_dict:
            return (sample,)
        return UNetSpatioTemporalConditionOutput(sample=sample)
