# ComfyUI wrapper nodes for [Mochi](https://github.com/genmoai/models) video generator


# WORK IN PROGRESS

https://github.com/user-attachments/assets/a714b70f-dcdb-4f91-8a3d-8da679a28d6e


## Requires flash_attn !
Not sure if this can be worked around, I compiled a wheel for my Windows setup (Python 3.12, torch 2.5.0+cu124) that worked for me:

https://huggingface.co/Kijai/Mochi_preview_comfy/blob/main/flash_attn-2.6.3-cp312-cp312-win_amd64.whl

Python 3.10 / CUDA 12.4 / Torch 2.4.1:

https://huggingface.co/Kijai/Mochi_preview_comfy/blob/main/flash_attn-2.6.3-cp310-cp310-win_amd64.whl

Other sources for pre-compiled wheels:

https://github.com/oobabooga/flash-attention/releases

Depending on frame count can fit under 20GB, VAE decoding is heavy and there is experimental tiled decoder (taken from CogVideoX -diffusers code) which allows higher frame counts, so far highest I've done is 97 with the default tile size 2x2 grid.

Models:

https://huggingface.co/Kijai/Mochi_preview_comfy/tree/main

model to: `ComfyUI/models/diffusion_models/mochi`

vae to: `ComfyUI/models/vae/mochi`

There is autodownload node (also will be normal loader node)
