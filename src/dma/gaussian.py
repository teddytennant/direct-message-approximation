"""Gaussian moment ↔ natural-parameter conversions.

Natural parameters of N(μ, σ²) are (τ, ρ) = (μ/σ², 1/σ²), so μ = τ/ρ and
σ² = 1/ρ. Variances and precisions are clipped to VAR_FLOOR to keep messages
proper; the paper formulas themselves are implemented without rewriting.
"""

from __future__ import annotations

import jax.numpy as jnp

VAR_FLOOR = 1e-12


def clip_var(var):
    return jnp.maximum(var, VAR_FLOOR)


def moments_to_natural(mu, var):
    """(μ, σ²) → (τ, ρ) = (μ/σ², 1/σ²)."""
    var = clip_var(var)
    rho = 1.0 / var
    tau = mu * rho
    return tau, rho


def natural_to_moments(tau, rho):
    """(τ, ρ) → (μ, σ²) = (τ/ρ, 1/ρ)."""
    rho = jnp.maximum(rho, VAR_FLOOR)
    var = 1.0 / rho
    mu = tau * var
    return mu, var


def multiply_natural(tau_a, rho_a, tau_b, rho_b):
    """Product of two Gaussians in natural parameters."""
    return tau_a + tau_b, rho_a + rho_b
