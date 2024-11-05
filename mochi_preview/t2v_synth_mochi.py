from typing import Dict, List, Optional, Union
from einops import rearrange

#temporary patch to fix torch compile bug in Windows
def patched_write_atomic(
    path_: str,
    content: Union[str, bytes],
    make_dirs: bool = False,
    encode_utf_8: bool = False,
) -> None:
    # Write into temporary file first to avoid conflicts between threads
    # Avoid using a named temporary file, as those have restricted permissions
    from pathlib import Path
    import os
    import shutil
    import threading
    assert isinstance(
        content, (str, bytes)
    ), "Only strings and byte arrays can be saved in the cache"
    path = Path(path_)
    if make_dirs:
        path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".{os.getpid()}.{threading.get_ident()}.tmp"
    write_mode = "w" if isinstance(content, str) else "wb"
    with tmp_path.open(write_mode, encoding="utf-8" if encode_utf_8 else None) as f:
        f.write(content)
    shutil.copy2(src=tmp_path, dst=path) #changed to allow overwriting cache files
    os.remove(tmp_path)
try:
    import torch._inductor.codecache
    torch._inductor.codecache.write_atomic = patched_write_atomic
except:
    pass

import torch
import torch.utils.data

from tqdm import tqdm
from comfy.utils import ProgressBar, load_torch_file
import comfy.model_management as mm 

from contextlib import nullcontext
try:
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    is_accelerate_available = True
except:
    is_accelerate_available = False
    pass

from .dit.joint_model.asymm_models_joint import AsymmDiTJoint

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

MAX_T5_TOKEN_LENGTH = 256

def fft(tensor):
    tensor_fft = torch.fft.fft2(tensor)
    tensor_fft_shifted = torch.fft.fftshift(tensor_fft)
    B, C, H, W = tensor.size()
    radius = min(H, W) // 5
            
    Y, X = torch.meshgrid(torch.arange(H), torch.arange(W))
    center_x, center_y = W // 2, H // 2
    mask = (X - center_x) ** 2 + (Y - center_y) ** 2 <= radius ** 2
    low_freq_mask = mask.unsqueeze(0).unsqueeze(0).to(tensor.device)
    high_freq_mask = ~low_freq_mask
            
    low_freq_fft = tensor_fft_shifted * low_freq_mask
    high_freq_fft = tensor_fft_shifted * high_freq_mask

    return low_freq_fft, high_freq_fft

class T2VSynthMochiModel:
    def __init__(
        self,
        *,
        device: torch.device,
        offload_device: torch.device,
        dit_checkpoint_path: str,
        weight_dtype: torch.dtype = torch.float8_e4m3fn,
        fp8_fastmode: bool = False,
        attention_mode: str = "sdpa",
        rms_norm_func: str = "default",
        compile_args: Optional[Dict] = None,
        cublas_ops: Optional[bool] = False,
    ):
        super().__init__()
        self.device = device
        self.weight_dtype = weight_dtype
        self.offload_device = offload_device

        logging.info("Initializing model...")
        with (init_empty_weights() if is_accelerate_available else nullcontext()):
            model = AsymmDiTJoint(
                depth=48,
                patch_size=2,
                num_heads=24,
                hidden_size_x=3072,
                hidden_size_y=1536,
                mlp_ratio_x=4.0,
                mlp_ratio_y=4.0,
                in_channels=12,
                qk_norm=True,
                qkv_bias=False,
                out_bias=True,
                patch_embed_bias=True,
                timestep_mlp_bias=True,
                timestep_scale=1000.0,
                t5_feat_dim=4096,
                t5_token_length=256,
                rope_theta=10000.0,
                attention_mode=attention_mode,
                rms_norm_func=rms_norm_func,
            )

        params_to_keep = {"t_embedder", "x_embedder", "pos_frequencies", "t5", "norm"}
        logging.info(f"Loading model state_dict from {dit_checkpoint_path}...")
        dit_sd = load_torch_file(dit_checkpoint_path)

        #comfy format
        prefix = "model.diffusion_model."
        first_key = next(iter(dit_sd), None)
        if first_key and first_key.startswith(prefix):
            new_dit_sd = {
                key[len(prefix):] if key.startswith(prefix) else key: value
                for key, value in dit_sd.items()
            }
            dit_sd = new_dit_sd
                
        if "gguf" in dit_checkpoint_path.lower():
            logging.info("Loading GGUF model state_dict...")
            from .. import mz_gguf_loader
            import importlib
            importlib.reload(mz_gguf_loader)
            with mz_gguf_loader.quantize_lazy_load():
                model = mz_gguf_loader.quantize_load_state_dict(model, dit_sd, device="cpu", cublas_ops=cublas_ops)
        elif is_accelerate_available:
            logging.info("Using accelerate to load and assign model weights to device...")
            for name, param in model.named_parameters():
                if not any(keyword in name for keyword in params_to_keep):
                    set_module_tensor_to_device(model, name, dtype=weight_dtype, device=self.device, value=dit_sd[name])
                else:
                    set_module_tensor_to_device(model, name, dtype=torch.bfloat16, device=self.device, value=dit_sd[name])
        else:
            logging.info("Loading state_dict without accelerate...")
            model.load_state_dict(dit_sd)
            for name, param in model.named_parameters():
                if not any(keyword in name for keyword in params_to_keep):
                    param.data = param.data.to(weight_dtype)
                else:
                    param.data = param.data.to(torch.bfloat16)
        
        if fp8_fastmode:
            from ..fp8_optimization import convert_fp8_linear
            convert_fp8_linear(model, torch.bfloat16)

        model = model.eval().to(self.device)

        #torch.compile
        if compile_args is not None:
            if compile_args["compile_dit"]:
                for i, block in enumerate(model.blocks):
                    model.blocks[i] = torch.compile(block, fullgraph=compile_args["fullgraph"], dynamic=compile_args["dynamic"], backend=compile_args["backend"])
            if compile_args["compile_final_layer"]:
                model.final_layer = torch.compile(model.final_layer, fullgraph=compile_args["fullgraph"], dynamic=compile_args["dynamic"], backend=compile_args["backend"])        

        self.dit = model

    def move_to_device_(self, sample):
        if isinstance(sample, dict):
            for key in sample.keys():
                if isinstance(sample[key], torch.Tensor):
                    sample[key] = sample[key].to(self.device, non_blocking=True)

    def run(self, args):
        torch.manual_seed(args["seed"])
        torch.cuda.manual_seed(args["seed"])

        generator = torch.Generator(device=torch.device("cpu"))
        generator.manual_seed(args["seed"])

        num_frames = args["num_frames"]
        height = args["height"]
        width = args["width"]
        in_samples = args["samples"]
        
        sample_steps = args["mochi_args"]["num_inference_steps"]
        cfg_schedule = args["mochi_args"].get("cfg_schedule")
        assert (
            len(cfg_schedule) == sample_steps
        ), f"cfg_schedule must have length {sample_steps}, got {len(cfg_schedule)}"
        sigma_schedule = args["mochi_args"].get("sigma_schedule")
        if sigma_schedule:
            assert (
                len(sigma_schedule) == sample_steps + 1
            ), f"sigma_schedule must have length {sample_steps + 1}, got {len(sigma_schedule)}"
        assert (num_frames - 1) % 6 == 0, f"t - 1 must be divisible by 6, got {num_frames - 1}"

        from ..latent_preview import prepare_callback
        callback = prepare_callback(self.dit, sample_steps)

        # create z
        spatial_downsample = 8
        temporal_downsample = 6
        in_channels = 12
        B = 1
        C = in_channels
        T = (num_frames - 1) // temporal_downsample + 1
        H = height // spatial_downsample
        W = width // spatial_downsample
        
        z = torch.randn(
            (B, C, T, H, W),
            device=torch.device("cpu"),
            generator=generator,
            dtype=torch.float32,
        ).to(self.device)
        if in_samples is not None:
            z = z * sigma_schedule[0] + (1 -sigma_schedule[0]) * in_samples.to(self.device)

        sample = {
            "y_mask": [args["positive_embeds"]["attention_mask"].to(self.device)],
            "y_feat": [args["positive_embeds"]["embeds"].to(self.device)],
        }
        sample_null = {
            "y_mask": [args["negative_embeds"]["attention_mask"].to(self.device)],
            "y_feat": [args["negative_embeds"]["embeds"].to(self.device)],
        }
        print(args["fastercache"])
        if args["fastercache"]:
            print("Using fastercache")
            self.fastercache_start_step = args["fastercache"]["start_step"]
            self.fastercache_lf_step = args["fastercache"]["lf_step"]
            self.fastercache_hf_step = args["fastercache"]["hf_step"]
        else:
            self.fastercache_start_step = 1000
        self.fastercache_counter = 0

        def model_fn(*, z, sigma, cfg_scale):  
            nonlocal sample, sample_null
            if cfg_scale != 1.0:
                if args["fastercache"]:
                    self.fastercache_counter+=1
                if self.fastercache_counter >= self.fastercache_start_step + 3 and self.fastercache_counter % 5 !=0:
                    out_cond = self.dit(
                        z, 
                        sigma,
                        **sample,
                        fastercache = args["fastercache"],
                        fastercache_counter=self.fastercache_counter)
                    
                    (bb, cc, tt, hh, ww) = out_cond.shape
                    cond = rearrange(out_cond, "B C T H W -> (B T) C H W", B=bb, C=cc, T=tt, H=hh, W=ww)
                    lf_c, hf_c = fft(cond.float())
                    if self.fastercache_counter <= self.fastercache_lf_step:
                        self.delta_lf = self.delta_lf * 1.1
                    if self.fastercache_counter >= self.fastercache_hf_step:
                        self.delta_hf = self.delta_hf * 1.1

                    new_hf_uc = self.delta_hf + hf_c
                    new_lf_uc = self.delta_lf + lf_c

                    combine_uc = new_lf_uc + new_hf_uc
                    combined_fft = torch.fft.ifftshift(combine_uc)
                    recovered_uncond = torch.fft.ifft2(combined_fft).real
                    recovered_uncond = rearrange(recovered_uncond.to(out_cond.dtype), "(B T) C H W -> B C T H W", B=bb, C=cc, T=tt, H=hh, W=ww)
                    
                    return recovered_uncond + cfg_scale * (out_cond - recovered_uncond)
                else:
                    out_cond = self.dit(
                        z, 
                        sigma, 
                        **sample,
                        fastercache = args["fastercache"],
                        fastercache_counter=self.fastercache_counter)
                    
                    out_uncond = self.dit(
                        z, 
                        sigma,  
                        **sample_null,
                        fastercache = args["fastercache"],
                        fastercache_counter=self.fastercache_counter)

                    if self.fastercache_counter >= self.fastercache_start_step + 1:
                        (bb, cc, tt, hh, ww) = out_cond.shape
                        cond = rearrange(out_cond.float(), "B C T H W -> (B T) C H W", B=bb, C=cc, T=tt, H=hh, W=ww)
                        uncond = rearrange(out_uncond.float(), "B C T H W -> (B T) C H W", B=bb, C=cc, T=tt, H=hh, W=ww)

                        lf_c, hf_c = fft(cond)
                        lf_uc, hf_uc = fft(uncond)

                        self.delta_lf = lf_uc - lf_c
                        self.delta_hf = hf_uc - hf_c
                        
                    return out_uncond + cfg_scale * (out_cond - out_uncond)
            else: #handle cfg 1.0
                out_cond = self.dit(
                        z, 
                        sigma,  
                        **sample,
                        fastercache = args["fastercache"],
                        fastercache_counter=self.fastercache_counter)
                return out_cond

                
        comfy_pbar = ProgressBar(sample_steps)

        if hasattr(self.dit, "cublas_half_matmul") and self.dit.cublas_half_matmul:
            autocast_dtype = torch.float16
        else:
            autocast_dtype = torch.bfloat16

        self.dit.to(self.device)

        with torch.autocast(mm.get_autocast_device(self.device), dtype=autocast_dtype):
            for i in tqdm(range(0, sample_steps), desc="Processing Samples", total=sample_steps):
                sigma = sigma_schedule[i]
                dsigma = sigma - sigma_schedule[i + 1]

                # `pred` estimates `z_0 - eps`.
                pred = model_fn(
                    z=z,
                    sigma=torch.full([B], sigma, device=z.device),
                    cfg_scale=cfg_schedule[i],
                )
                z = z + dsigma * pred.to(z)
                if callback is not None:
                    callback(i, z.detach()[0].permute(1,0,2,3), None, sample_steps)
                else:
                    comfy_pbar.update(1)

        if args["fastercache"] is not None:
            for block in self.dit.blocks:
                if (hasattr, block, "cached_x_attention") and block.cached_x_attention is not None:
                    block.cached_x_attention = None
                    block.cached_y_attention = None
       
        self.dit.to(self.offload_device)
        mm.soft_empty_cache()
        logging.info(f"samples shape: {z.shape}")
        return z
