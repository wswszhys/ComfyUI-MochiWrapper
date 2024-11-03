from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def pool_tokens(x: torch.Tensor, mask: torch.Tensor, *, keepdim=False) -> torch.Tensor:
    """
    Pool tokens in x using mask.

    NOTE: We assume x does not require gradients.

    Args:
        x: (B, L, D) tensor of tokens.
        mask: (B, L) boolean tensor indicating which tokens are not padding.

    Returns:
        pooled: (B, D) tensor of pooled tokens.
    """
    assert x.size(1) == mask.size(1)  # Expected mask to have same length as tokens.
    assert x.size(0) == mask.size(0)  # Expected mask to have same batch size as tokens.
    mask = mask[:, :, None].to(dtype=x.dtype)
    mask = mask / mask.sum(dim=1, keepdim=True).clamp(min=1)
    pooled = (x * mask).sum(dim=1, keepdim=keepdim)
    return pooled
    