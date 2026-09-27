"""Direct Message Approximation (DMA) messages from arXiv:2609.29466."""

from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

from dma.gaussian import VAR_FLOOR, moments_to_natural, natural_to_moments
from dma.product import product_backward, product_forward
from dma.relu import relu_backward, relu_forward
from dma.sum import sum_cavity, sum_forward
from dma.bnn import TinyBNN

__all__ = [
    "VAR_FLOOR",
    "moments_to_natural",
    "natural_to_moments",
    "product_forward",
    "product_backward",
    "relu_forward",
    "relu_backward",
    "sum_forward",
    "sum_cavity",
    "TinyBNN",
]
