"""Leaky-ReLU factor δ(y − ReLU_α(x)): Propositions 3.4 and 3.5.

ReLU_α(x) = max(0, x) + α min(0, x) with α > 0.
Outgoing messages do not depend on the incoming message on the same edge.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax.scipy.stats import norm

from dma.gaussian import clip_var


def _unit_gaussian(u):
    """φ(u) = N(u; 0, 1) density, Φ(u) = N(u; 0, 1) cdf."""
    phi = norm.pdf(u)
    Phi = norm.cdf(u)
    return phi, Phi


def relu_forward(mu_x, var_x, alpha):
    """Forward message to Y. Proposition 3.4 / eq. (10).

    X ~ N(μx, σx²), u := μx/σx, P := Φ(u), ϕ := φ(u),
        A  := α + (1 − α) P
        B  := α² + (1 − α²) P
        mY = μx A + (1 − α) σx ϕ
        s²Y = (μx² + σx²) B + (1 − α²) μx σx ϕ − mY²
    Does not use an incoming message on Y. α = 1 (identity) recovers Y = X.
    """
    var_x = clip_var(var_x)
    sigma_x = jnp.sqrt(var_x)
    u = mu_x / sigma_x
    phi, P = _unit_gaussian(u)
    A = alpha + (1.0 - alpha) * P
    B = alpha ** 2 + (1.0 - alpha ** 2) * P
    mY = mu_x * A + (1.0 - alpha) * sigma_x * phi
    s2Y = (mu_x ** 2 + var_x) * B + (1.0 - alpha ** 2) * mu_x * sigma_x * phi - mY ** 2
    return mY, clip_var(s2Y)


def relu_backward(mu_y, var_y, alpha):
    """Backward message to X. Proposition 3.5 / eqs. (11)–(12).

    Y ~ N(μy, σy²), v := μy/σy, P := Φ(v), Q := 1 − P, ϕ := φ(v),
        C̃ := α P + Q
        mX  = [μy (α² P + Q) + (α² − 1) σy ϕ] / (α C̃)
        s²X = [(μy² + σy²)(α³ P + Q) + (α³ − 1) μy σy ϕ] / (α² C̃) − mX²
    Does not use an incoming message on X. α = 1 recovers X = Y.
    """
    var_y = clip_var(var_y)
    sigma_y = jnp.sqrt(var_y)
    v = mu_y / sigma_y
    phi, P = _unit_gaussian(v)
    Q = 1.0 - P
    C_tilde = alpha * P + Q
    mX = (mu_y * (alpha ** 2 * P + Q) + (alpha ** 2 - 1.0) * sigma_y * phi) / (
        alpha * C_tilde
    )
    s2X = (
        (mu_y ** 2 + var_y) * (alpha ** 3 * P + Q)
        + (alpha ** 3 - 1.0) * mu_y * sigma_y * phi
    ) / (alpha ** 2 * C_tilde) - mX ** 2
    return mX, clip_var(s2X)
