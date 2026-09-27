"""Linear sum factor: exact Gaussian messages (means and variances add).

Backward to each summand is Gaussian deconvolution as in Algorithm 1:
    μ_cav  = μ_bwd − μ_zi + μ_zij
    σ²_cav = σ²_bwd + σ²_zi − σ²_zij
"""

from __future__ import annotations

import jax.numpy as jnp

from dma.gaussian import clip_var


def sum_forward(mu_terms, var_terms, axis=-1):
    """Forward: μ_zi = Σ_j μ_zij, σ²_zi = Σ_j σ²_zij."""
    return jnp.sum(mu_terms, axis=axis), jnp.sum(var_terms, axis=axis)


def sum_cavity(mu_bwd, var_bwd, mu_zi, var_zi, mu_zij, var_zij):
    """Cavity message to one summand z_ij given a backward message on z_i."""
    mu_cav = mu_bwd - mu_zi + mu_zij
    var_cav = var_bwd + var_zi - var_zij
    return mu_cav, clip_var(var_cav)
