from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import (
    FeedForward,
    PatchEmbed,
    TimestepEmbedder,
)


from .mod_rmsnorm import modulated_rmsnorm
from .residual_tanh_gated_rmsnorm import residual_tanh_gated_rmsnorm
from .rope_mixed import compute_mixed_rotation, create_position_matrix
from .temporal_rope import apply_rotary_emb_qk_real
from .utils import pool_tokens, modulate

try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_IS_AVAILABLE = True
except ImportError:
    FLASH_ATTN_IS_AVAILABLE = False
try:
    from sageattention import sageattn
    SAGEATTN_IS_AVAILABLE = True
except ImportError:
    SAGEATTN_IS_AVAILABLE = False

from torch.nn.attention import sdpa_kernel, SDPBackend

backends = []
backends.append(SDPBackend.CUDNN_ATTENTION)
backends.append(SDPBackend.EFFICIENT_ATTENTION)
backends.append(SDPBackend.MATH)

import comfy.model_management as mm
from comfy.ldm.modules.attention import optimized_attention

class AttentionPool(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        output_dim: int = None,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            spatial_dim (int): Number of tokens in sequence length.
            embed_dim (int): Dimensionality of input tokens.
            num_heads (int): Number of attention heads.
            output_dim (int): Dimensionality of output tokens. Defaults to embed_dim.
        """
        super().__init__()
        self.num_heads = num_heads
        self.to_kv = nn.Linear(embed_dim, 2 * embed_dim, device=device)
        self.to_q = nn.Linear(embed_dim, embed_dim, device=device)
        self.to_out = nn.Linear(embed_dim, output_dim or embed_dim, device=device)

    def forward(self, x, mask):
        """
        Args:
            x (torch.Tensor): (B, L, D) tensor of input tokens.
            mask (torch.Tensor): (B, L) boolean tensor indicating which tokens are not padding.

        NOTE: We assume x does not require gradients.

        Returns:
            x (torch.Tensor): (B, D) tensor of pooled tokens.
        """
        D = x.size(2)

        # Construct attention mask, shape: (B, 1, num_queries=1, num_keys=1+L).
        attn_mask = mask[:, None, None, :].bool()  # (B, 1, 1, L).
        attn_mask = F.pad(attn_mask, (1, 0), value=True)  # (B, 1, 1, 1+L).

        # Average non-padding token features. These will be used as the query.
        x_pool = pool_tokens(x, mask, keepdim=True)  # (B, 1, D)

        # Concat pooled features to input sequence.
        x = torch.cat([x_pool, x], dim=1)  # (B, L+1, D)

        # Compute queries, keys, values. Only the mean token is used to create a query.
        kv = self.to_kv(x)  # (B, L+1, 2 * D)
        q = self.to_q(x[:, 0])  # (B, D)

        # Extract heads.
        head_dim = D // self.num_heads
        kv = kv.unflatten(2, (2, self.num_heads, head_dim))  # (B, 1+L, 2, H, head_dim)
        kv = kv.transpose(1, 3)  # (B, H, 2, 1+L, head_dim)
        k, v = kv.unbind(2)  # (B, H, 1+L, head_dim)
        q = q.unflatten(1, (self.num_heads, head_dim))  # (B, H, head_dim)
        q = q.unsqueeze(2)  # (B, H, 1, head_dim)

        # Compute attention.
        x = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.0
        )  # (B, H, 1, head_dim)

        # Concatenate heads and run output.
        x = x.squeeze(2).flatten(1, 2)  # (B, D = H * head_dim)
        x = self.to_out(x)
        return x
    
#region Attention
class AsymmetricAttention(nn.Module):
    def __init__(
        self,
        dim_x: int,
        dim_y: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        update_y: bool = True,
        out_bias: bool = True,
        attend_to_padding: bool = False,
        softmax_scale: Optional[float] = None,
        device: Optional[torch.device] = None,
        attention_mode: str = "sdpa",
        rms_norm_func: bool = False,
        
    ):
        super().__init__()
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.num_heads = num_heads
        self.head_dim = dim_x // num_heads
        self.attn_drop = attn_drop
        self.update_y = update_y
        self.attend_to_padding = attend_to_padding
        self.softmax_scale = softmax_scale
        self.attention_mode = attention_mode
        self.device = device
        if dim_x % num_heads != 0:
            raise ValueError(
                f"dim_x={dim_x} should be divisible by num_heads={num_heads}"
            )

        # Input layers.
        self.qkv_bias = qkv_bias
        self.qkv_x = nn.Linear(dim_x, 3 * dim_x, bias=qkv_bias, device=device)
        # Project text features to match visual features (dim_y -> dim_x)
        self.qkv_y = nn.Linear(dim_y, 3 * dim_x, bias=qkv_bias, device=device)

        # Query and key normalization for stability.
        #assert qk_norm
        if rms_norm_func == "flash_attn_triton": #use the same rms_norm_func
            try:
                from flash_attn.ops.triton.layer_norm import RMSNorm as FlashTritonRMSNorm #slightly faster
                @torch.compiler.disable() #cause NaNs when compiled for some reason
                class RMSNorm(FlashTritonRMSNorm):
                    pass
            except:
                raise ImportError("Flash Triton RMSNorm not available.")
        elif rms_norm_func == "flash_attn":
            try:
                from flash_attn.ops.rms_norm import RMSNorm as FlashRMSNorm #slightly faster
                @torch.compiler.disable() #cause NaNs when compiled for some reason
                class RMSNorm(FlashRMSNorm):
                    pass
            except:
                raise ImportError("Flash RMSNorm not available.")
        elif rms_norm_func == "apex":
            from apex.normalization import FusedRMSNorm as ApexRMSNorm
            class RMSNorm(ApexRMSNorm):
                pass
        else:
            from .layers import RMSNorm
        norm_kwargs = {}
        if rms_norm_func != "apex":
            norm_kwargs['device'] = device

        self.q_norm_x = RMSNorm(self.head_dim, **norm_kwargs)
        self.k_norm_x = RMSNorm(self.head_dim, **norm_kwargs)
        self.q_norm_y = RMSNorm(self.head_dim, **norm_kwargs)
        self.k_norm_y = RMSNorm(self.head_dim, **norm_kwargs)

        # Output layers. y features go back down from dim_x -> dim_y.
        self.proj_x = nn.Linear(dim_x, dim_x, bias=out_bias, device=device)
        self.proj_y = (
            nn.Linear(dim_x, dim_y, bias=out_bias, device=device)
            if update_y
            else nn.Identity()
        )
    
    def flash_attention(self, q, k ,v):
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        b, _, _, dim_head = q.shape
        with torch.autocast(mm.get_autocast_device(self.device), enabled=False):
            out: torch.Tensor = flash_attn_func( #q: (batch_size, seqlen, nheads, headdim)
                q, k, v,
                dropout_p=0.0,
                softmax_scale=self.softmax_scale,
            )  # (total, local_heads, head_dim)
            out = out.permute(0, 2, 1, 3)
            return out.transpose(1, 2).reshape(b, -1, self.num_heads * dim_head)
        
    def sdpa_attention(self, q, k, v):
        b, _, _, dim_head = q.shape
        with torch.autocast(mm.get_autocast_device(self.device), enabled=False):
            with sdpa_kernel(backends):
                out = F.scaled_dot_product_attention(
                    q, 
                    k, 
                    v, 
                    attn_mask=None, 
                    dropout_p=0.0, 
                    is_causal=False
                    )
                return out.transpose(1, 2).reshape(b, -1, self.num_heads * dim_head)
        
    def sage_attention(self, q, k, v):
        b, _, _, dim_head = q.shape
        with torch.autocast(mm.get_autocast_device(self.device), enabled=False):
            out = sageattn(
                q, 
                k, 
                v, 
                attn_mask=None, 
                dropout_p=0.0, 
                is_causal=False
                )
            return out.transpose(1, 2).reshape(b, -1, self.num_heads * dim_head)
        
    def comfy_attention(self, q, k, v):
        with torch.autocast(mm.get_autocast_device(self.device), enabled=False):
            out = optimized_attention(
                q, 
                k, 
                v, 
                heads = self.num_heads,
                skip_reshape=True
                )
            return out

    def run_attention(
        self,
        q,
        k,
        v, 
    ):      
        if self.attention_mode == "flash_attn":
            out = self.flash_attention(q, k ,v)
        elif self.attention_mode == "sdpa":
            out = self.sdpa_attention(q, k, v)
        elif self.attention_mode == "sage_attn":
            out = self.sage_attention(q, k, v)
        elif self.attention_mode == "comfy":
            out = self.comfy_attention(q, k, v)
        return out

    def forward(
        self,
        x: torch.Tensor,  # (B, N, dim_x)
        y: torch.Tensor,  # (B, L, dim_y)
        *,
        scale_x: torch.Tensor,  # (B, dim_x), modulation for pre-RMSNorm.
        scale_y: torch.Tensor,  # (B, dim_y), modulation for pre-RMSNorm.
        num_tokens,
        **rope_rotation,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        rope_cos = rope_rotation.get("rope_cos")
        rope_sin = rope_rotation.get("rope_sin")
        
        # Pre-norm for visual features
        x = modulated_rmsnorm(x, scale_x)  # (B, M, dim_x) where M = N / cp_group_size
        # Process text features
        y = modulated_rmsnorm(y, scale_y)  # (B, L, dim_y)
        q_y, k_y, v_y = self.qkv_y(y).view(y.shape[0], y.shape[1], 3, self.num_heads, -1).unbind(2)  # (B, N, local_h, head_dim)

        q_y = self.q_norm_y(q_y)
        k_y = self.k_norm_y(k_y)

        # Split qkv_x into q, k, v
        q_x, k_x, v_x = self.qkv_x(x).view(x.shape[0], x.shape[1], 3, self.num_heads, -1).unbind(2)  # (B, N, local_h, head_dim)
        q_x = self.q_norm_x(q_x)
        q_x = apply_rotary_emb_qk_real(q_x, rope_cos, rope_sin)
        k_x = self.k_norm_x(k_x)
        k_x = apply_rotary_emb_qk_real(k_x, rope_cos, rope_sin)

        q = torch.cat([q_x, q_y[:, :num_tokens]], dim=1).transpose(1, 2)
        k = torch.cat([k_x, k_y[:, :num_tokens]], dim=1).transpose(1, 2)
        v = torch.cat([v_x, v_y[:, :num_tokens]], dim=1).transpose(1, 2)

        xy = self.run_attention(q, k, v)

        x, y = torch.tensor_split(xy, (q_x.shape[1],), dim=1)
        x = self.proj_x(x)
        o = torch.zeros(y.shape[0], q_y.shape[1], y.shape[-1], device=y.device, dtype=y.dtype)
        o[:, :y.shape[1]] = y

        y = self.proj_y(o)
        return x, y
    
#region Blocks
class AsymmetricJointBlock(nn.Module):
    def __init__(
        self,
        hidden_size_x: int,
        hidden_size_y: int,
        num_heads: int,
        *,
        mlp_ratio_x: float = 8.0,  # Ratio of hidden size to d_model for MLP for visual tokens.
        mlp_ratio_y: float = 4.0,  # Ratio of hidden size to d_model for MLP for text tokens.
        update_y: bool = True,  # Whether to update text tokens in this block.
        device: Optional[torch.device] = None,
        attention_mode: str = "sdpa",
        rms_norm_func: str = "default",
        **block_kwargs,
    ):
        super().__init__()
        self.update_y = update_y
        self.hidden_size_x = hidden_size_x
        self.hidden_size_y = hidden_size_y
        self.attention_mode = attention_mode
        self.mod_x = nn.Linear(hidden_size_x, 4 * hidden_size_x, device=device)
        if self.update_y:
            self.mod_y = nn.Linear(hidden_size_x, 4 * hidden_size_y, device=device)
        else:
            self.mod_y = nn.Linear(hidden_size_x, hidden_size_y, device=device)

        self.cached_x_attention = [None, None]
        self.cached_y_attention = [None, None]
        
        # Self-attention:
        self.attn = AsymmetricAttention(
            hidden_size_x,
            hidden_size_y,
            num_heads=num_heads,
            update_y=update_y,
            device=device,
            attention_mode=attention_mode,
            rms_norm_func=rms_norm_func,
            **block_kwargs,
        )

        # MLP.
        mlp_hidden_dim_x = int(hidden_size_x * mlp_ratio_x)
        #assert mlp_hidden_dim_x == int(1536 * 8)
        self.mlp_x = FeedForward(
            in_features=hidden_size_x,
            hidden_size=mlp_hidden_dim_x,
            multiple_of=256,
            ffn_dim_multiplier=None,
            device=device,
        )

        # MLP for text not needed in last block.
        if self.update_y:
            mlp_hidden_dim_y = int(hidden_size_y * mlp_ratio_y)
            self.mlp_y = FeedForward(
                in_features=hidden_size_y,
                hidden_size=mlp_hidden_dim_y,
                multiple_of=256,
                ffn_dim_multiplier=None,
                device=device,
            )

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        y: torch.Tensor,
        fastercache_counter: Optional[int] = 0,
        fastercache_start_step: Optional[int]  = 1000,
        fastercache_device: Optional[torch.device] = None,
        **attn_kwargs,
    ):
        """Forward pass of a block.

        Args:
            x: (B, N, dim) tensor of visual tokens
            c: (B, dim) tensor of conditioned features
            y: (B, L, dim) tensor of text tokens
            num_frames: Number of frames in the video. N = num_frames * num_spatial_tokens

        Returns:
            x: (B, N, dim) tensor of visual tokens after block
            y: (B, L, dim) tensor of text tokens after block
        """
        N = x.size(1)

        c = F.silu(c)
        mod_x = self.mod_x(c)
        scale_msa_x, gate_msa_x, scale_mlp_x, gate_mlp_x = mod_x.chunk(4, dim=1)

        mod_y = self.mod_y(c)
        if self.update_y:
            scale_msa_y, gate_msa_y, scale_mlp_y, gate_mlp_y = mod_y.chunk(4, dim=1)
        else:
            scale_msa_y = mod_y
        
        #region fastercache
        B = x.shape[0]
        #print("x", x.shape) #([1, 9540, 3072])
        if fastercache_counter >= fastercache_start_step + 3 and fastercache_counter%3!=0 and self.cached_x_attention[-1].shape[0] >= B:
            x_attn = (
                self.cached_x_attention[1][:B] + 
                (self.cached_x_attention[1][:B] - self.cached_x_attention[0][:B]) 
                * 0.3
                ).to(x.device, non_blocking=True)
            y_attn = (
                self.cached_y_attention[1][:B] + 
                (self.cached_y_attention[1][:B] - self.cached_y_attention[0][:B]) 
                * 0.3
                ).to(x.device, non_blocking=True)
        else:
            # Self-attention block.
            x_attn, y_attn = self.attn(
                x,
                y,
                scale_x=scale_msa_x,
                scale_y=scale_msa_y,
                **attn_kwargs,
            )
            if fastercache_counter == fastercache_start_step:
                self.cached_x_attention = [x_attn.to(fastercache_device), x_attn.to(fastercache_device)]
                self.cached_y_attention = [y_attn.to(fastercache_device), y_attn.to(fastercache_device)]    
            elif fastercache_counter > fastercache_start_step:
                self.cached_x_attention[-1].copy_(x_attn.to(fastercache_device))
                self.cached_y_attention[-1].copy_(y_attn.to(fastercache_device))

        #assert x_attn.size(1) == N
        x = residual_tanh_gated_rmsnorm(x, x_attn, gate_msa_x)
        if self.update_y:
            y = residual_tanh_gated_rmsnorm(y, y_attn, gate_msa_y)
       
        # MLP block.
        x = self.ff_block_x(x, scale_mlp_x, gate_mlp_x)
        if self.update_y:
            y = self.ff_block_y(y, scale_mlp_y, gate_mlp_y)
      
        return x, y

    def ff_block_x(self, x, scale_x, gate_x):
        x_mod = modulated_rmsnorm(x, scale_x)
        x_res = self.mlp_x(x_mod)
        x = residual_tanh_gated_rmsnorm(x, x_res, gate_x)  # Sandwich norm
        return x

    def ff_block_y(self, y, scale_y, gate_y):
        y_mod = modulated_rmsnorm(y, scale_y)
        y_res = self.mlp_y(y_mod)
        y = residual_tanh_gated_rmsnorm(y, y_res, gate_y)  # Sandwich norm
        return y


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(
        self,
        hidden_size,
        patch_size,
        out_channels,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.norm_final = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, device=device
        )
        self.mod = nn.Linear(hidden_size, 2 * hidden_size, device=device)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, device=device
        )

    def forward(self, x, c):
        c = F.silu(c)
        shift, scale = self.mod(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

#region Model
class AsymmDiTJoint(nn.Module):
    """
    Diffusion model with a Transformer backbone.

    Ingests text embeddings instead of a label.
    """

    def __init__(
        self,
        *,
        patch_size=2,
        in_channels=4,
        hidden_size_x=1152,
        hidden_size_y=1152,
        depth=48,
        num_heads=16,
        mlp_ratio_x=8.0,
        mlp_ratio_y=4.0,
        t5_feat_dim: int = 4096,
        t5_token_length: int = 256,
        patch_embed_bias: bool = True,
        timestep_mlp_bias: bool = True,
        timestep_scale: Optional[float] = None,
        use_extended_posenc: bool = False,
        rope_theta: float = 10000.0,
        device: Optional[torch.device] = None,
        attention_mode: str = "sdpa",
        rms_norm_func: str = "default",
        **block_kwargs,
    ):
        super().__init__()


        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size_x = hidden_size_x
        self.hidden_size_y = hidden_size_y
        self.head_dim = (
            hidden_size_x // num_heads
        )  # Head dimension and count is determined by visual.
        self.use_extended_posenc = use_extended_posenc
        self.t5_token_length = t5_token_length
        self.t5_feat_dim = t5_feat_dim
        self.rope_theta = (
            rope_theta  # Scaling factor for frequency computation for temporal RoPE.
        )

        self.x_embedder = PatchEmbed(
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=hidden_size_x,
            bias=patch_embed_bias,
            device=device,
        )
        # Conditionings
        # Timestep
        self.t_embedder = TimestepEmbedder(
            hidden_size_x, bias=timestep_mlp_bias, timestep_scale=timestep_scale
        )

        # Caption Pooling (T5)
        self.t5_y_embedder = AttentionPool(
            t5_feat_dim, num_heads=8, output_dim=hidden_size_x, device=device
        )

        # Dense Embedding Projection (T5)
        self.t5_yproj = nn.Linear(
            t5_feat_dim, hidden_size_y, bias=True, device=device
        )

        # Initialize pos_frequencies as an empty parameter.
        self.pos_frequencies = nn.Parameter(
            torch.empty(3, self.num_heads, self.head_dim // 2, device=device)
        )

        # for depth 48:
        #  b =  0: AsymmetricJointBlock, update_y=True
        #  b =  1: AsymmetricJointBlock, update_y=True
        #  ...
        #  b = 46: AsymmetricJointBlock, update_y=True
        #  b = 47: AsymmetricJointBlock, update_y=False. No need to update text features.
        blocks = []
        for b in range(depth):
            # Joint multi-modal block
            update_y = b < depth - 1
            block = AsymmetricJointBlock(
                hidden_size_x,
                hidden_size_y,
                num_heads,
                mlp_ratio_x=mlp_ratio_x,
                mlp_ratio_y=mlp_ratio_y,
                update_y=update_y,
                device=device,
                attention_mode=attention_mode,
                rms_norm_func=rms_norm_func,
                **block_kwargs,
            )

            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)

        self.final_layer = FinalLayer(
            hidden_size_x, patch_size, self.out_channels, device=device
        )

    def embed_x(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C=12, T, H, W) tensor of visual tokens

        Returns:
            x: (B, C=3072, N) tensor of visual tokens with positional embedding.
        """
        return self.x_embedder(x)  # Convert BcTHW to BCN

    def prepare(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        t5_feat: torch.Tensor,
        t5_mask: torch.Tensor,
    ):
        """Prepare input and conditioning embeddings."""
        #("X", x.shape)
        # Visual patch embeddings with positional encoding.
        T, H, W = x.shape[-3:]
        pH, pW = H // self.patch_size, W // self.patch_size
        x = self.embed_x(x)  # (B, N, D), where N = T * H * W / patch_size ** 2
        #assert x.ndim == 3

        # Construct position array of size [N, 3].
        # pos[:, 0] is the frame index for each location,
        # pos[:, 1] is the row index for each location, and
        # pos[:, 2] is the column index for each location.
        pH, pW = H // self.patch_size, W // self.patch_size
        N = T * pH * pW
        #assert x.size(1) == N
        pos = create_position_matrix(T, pH=pH, pW=pW, device=x.device, dtype=torch.float32)  # (N, 3)
        rope_cos, rope_sin = compute_mixed_rotation(freqs=self.pos_frequencies, pos=pos)  # Each are (N, num_heads, dim // 2)

        # Global vector embedding for conditionings.
        c_t = self.t_embedder(1 - sigma)  # (B, D)

        # Pool T5 tokens using attention pooler
        # Note y_feat[1] contains T5 token features.
        t5_y_pool = self.t5_y_embedder(t5_feat, t5_mask)  # (B, D)

        c = c_t + t5_y_pool

        y_feat = self.t5_yproj(t5_feat)  # (B, L, t5_feat_dim) --> (B, L, D)

        return x, c, y_feat, rope_cos, rope_sin

    def forward(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        y_feat: List[torch.Tensor],
        y_mask: List[torch.Tensor],
        rope_cos: torch.Tensor = None,
        rope_sin: torch.Tensor = None,
        fastercache: Optional[Dict] = None,
        fastercache_counter: Optional[int]=0,
    ):
        """Forward pass of DiT.

        Args:
            x: (B, C, T, H, W) tensor of spatial inputs (images or latent representations of images)
            sigma: (B,) tensor of noise standard deviations
            y_feat: List((B, L, y_feat_dim) tensor of caption token features. For SDXL text encoders: L=77, y_feat_dim=2048)
            y_mask: List((B, L) boolean tensor indicating which tokens are not padding)
        """
        B, _, T, H, W = x.shape

        # Use EFFICIENT_ATTENTION backend for T5 pooling, since we have a mask.
        # Have to call sdpa_kernel outside of a torch.compile region.
        num_tokens = max(1, torch.sum(y_mask[0]).item())
        with sdpa_kernel(backends):
            x, c, y_feat, rope_cos, rope_sin = self.prepare(
                x, sigma, y_feat[0], y_mask[0]
            )
        del y_mask

        if fastercache is not None:
            fastercache_start_step = fastercache["start_step"]
            fastercache_device = fastercache["cache_device"]
        else:
            fastercache_start_step = 1000
            fastercache_device = None
        #print(fastercache_counter)
        
        for i, block in enumerate(self.blocks):
            x, y_feat = block(
                x,
                c,
                y_feat,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                num_tokens=num_tokens,
                fastercache_counter = fastercache_counter,
                fastercache_start_step = fastercache_start_step,
                fastercache_device = fastercache_device,

                )  # (B, M, D), (B, L, D)
        del y_feat  # Final layers don't use dense text features.

        x = self.final_layer(x, c)  # (B, M, patch_size ** 2 * out_channels)

        hp = H // self.patch_size
        wp = W // self.patch_size
        p1 = self.patch_size
        p2 = self.patch_size
        c = self.out_channels

        x = x.view(B, T, hp, wp, p1, p2, c)
        x = x.permute(0, 6, 1, 2, 4, 3, 5)
        x = x.reshape(B, c, T, hp * p1, wp * p2)

        return x
