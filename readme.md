# ComfyUI wrapper nodes for [Mochi](https://github.com/genmoai/models) video generator


# WORK IN PROGRESS

## Requires flash_attn !
Not sure if this can be worked around, I compiled a wheel for my Windows setup (Python 3.12, torch 2.5.0+cu124) that worked for me:

https://huggingface.co/Kijai/Mochi_preview_comfy/blob/main/flash_attn-2.6.3-cp312-torch250cu125-win_amd64.whl

Depending on frame count can fit under 20GB, VAE decoding is heavy and there is experimental tiled decoder (taken from CogVideoX -diffusers code) which allows higher frame counts, so far highest I've done is 97 with the default tile size 2x2 grid.

Models:

https://huggingface.co/Kijai/Mochi_preview_comfy/tree/main

model to: `ComfyUI/models/diffusion_models/mochi`

vae to: `ComfyUI/models/vae/mochi`

There is autodownload node (also will be normal loader node)
