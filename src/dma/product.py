"""Product factor δ(z − x y): Propositions 3.1 and 3.2 of arXiv:2609.29466.

Outgoing messages do not depend on the incoming message on the same edge
(DMA: no cavity division inside the factor).

Theorem 3.3's O(1/r²) KL bound vs a truncated reference is not checked here;
that needs the truncated-ratio reference of Appendix B.5.
"""

from __future__ import annotations

import jax.numpy as jnp

from dma.gaussian import moments_to_natural, natural_to_moments


def product_forward(mu_x, var_x, mu_y, var_y):
    """Forward message to Z. Proposition 3.1 / eq. (6).

    Independent X ~ N(μx, σx²), Y ~ N(μy, σy²):
        mz  = μx μy
        s²z = σx² σy² + μx² σy² + μy² σx²
    The message to Z is N(mz, s²z). Does not use an incoming message on Z.
    """
    mz = mu_x * mu_y
    sz2 = var_x * var_y + (mu_x ** 2) * var_y + (mu_y ** 2) * var_x
    return mz, sz2


def product_backward(tau_y, rho_y, tau_z, rho_z):
    """Backward message to X in natural parameters. Proposition 3.2 / eq. (7).

    Incoming Y ~ N(τy/ρy, 1/ρy), Z ~ N(τz/ρz, 1/ρz):
        r     := τy⁴ ρz / (ρy (τy² + ρy))
        D     := τz² ρy + τy² ρz + ρy ρz
        τ̂x    = (τy r τz) / D
        ρ̂x    = r² / D
    Message to Y by swapping x ↔ y. Does not use an incoming message on X.

    Dirac-Y (ρy → ∞) recovers X = Z / ȳ, i.e. mean μz/ȳ and variance σz²/ȳ².
    """
    r = (tau_y ** 4) * rho_z / (rho_y * (tau_y ** 2 + rho_y))
    D = (tau_z ** 2) * rho_y + (tau_y ** 2) * rho_z + rho_y * rho_z
    tau_hat_x = (tau_y * r * tau_z) / D
    rho_hat_x = (r ** 2) / D
    return tau_hat_x, rho_hat_x


def product_backward_moments(mu_y, var_y, mu_z, var_z):
    """Proposition 3.2 in moment parameters: message to X given Y and Z."""
    tau_y, rho_y = moments_to_natural(mu_y, var_y)
    tau_z, rho_z = moments_to_natural(mu_z, var_z)
    tau_x, rho_x = product_backward(tau_y, rho_y, tau_z, rho_z)
    return natural_to_moments(tau_x, rho_x)
