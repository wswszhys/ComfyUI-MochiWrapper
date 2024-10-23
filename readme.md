# ComfyUI wrapper nodes for [Mochi](https://github.com/genmoai/models) video generator


# WORK IN PROGRESS

## Requires flash_attn !

Depending on frame count can fit under 20GB, VAE decoding is heavy and there is experimental tiled decoder (taken from CogVideoX -diffusers code) which allows higher frame counts, so far highest I've done is 97 with the default tile size 2x2 grid.

Models:

https://huggingface.co/Kijai/Mochi_preview_comfy/tree/main

model to: `ComfyUI/models/diffusion_models/mochi`

vae to: `ComfyUI/models/vae/mochi`

There is autodownload node (also will be normal loader node)
