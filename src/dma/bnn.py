"""Tiny 1-hidden-layer scalar BNN, one-sweep DMA (Algorithm 1, simplified).

Architecture (d_in = 1, d_out = 1, d_hidden small):
    x (point mass) → product+sum with W1 → leaky-ReLU → product+sum with W2
    → Gaussian likelihood N(y; z2, β²).

Weights have Gaussian beliefs N(μ, σ0²) with He-style mean init N(0, 1/fan_in).
One forward sweep then one backward sweep per example (Corollary 2.5: no EP
inner loop). Product backward updates weight beliefs; messages have
non-negative precision (Corollary 2.6). Mini-batch cavity division is omitted.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from dma.gaussian import moments_to_natural, multiply_natural, natural_to_moments
from dma.product import product_backward, product_forward
from dma.relu import relu_backward, relu_forward
from dma.sum import sum_cavity, sum_forward


class TinyBNN:
    """1-hidden-layer DMA network for scalar regression."""

    def __init__(
        self,
        d_hidden=2,
        alpha=0.2,
        beta=0.2,
        sigma0=1.0,
        key=None,
    ):
        self.d_hidden = int(d_hidden)
        self.alpha = jnp.asarray(alpha)
        self.beta2 = jnp.asarray(beta) ** 2
        self.sigma0 = float(sigma0)
        if key is None:
            key = jax.random.PRNGKey(0)
        k1, k2 = jax.random.split(key)
        # He-style: μ ~ N(0, 1/fan_in), prior variance σ0².
        self.mu_w1 = jax.random.normal(k1, (self.d_hidden,)) * jnp.sqrt(1.0)
        self.var_w1 = jnp.full((self.d_hidden,), self.sigma0 ** 2)
        self.mu_w2 = jax.random.normal(k2, (self.d_hidden,)) * jnp.sqrt(
            1.0 / self.d_hidden
        )
        self.var_w2 = jnp.full((self.d_hidden,), self.sigma0 ** 2)

    def _forward_cache(self, x):
        """Forward sweep; x is a scalar (point-mass input, δ = 0)."""
        mu_x = jnp.asarray(x, dtype=jnp.float64)
        var_x = jnp.asarray(0.0, dtype=jnp.float64)

        mu_z1, var_z1 = product_forward(self.mu_w1, self.var_w1, mu_x, var_x)
        mu_h, var_h = relu_forward(mu_z1, var_z1, self.alpha)
        mu_z2j, var_z2j = product_forward(self.mu_w2, self.var_w2, mu_h, var_h)
        mu_z2, var_z2 = sum_forward(mu_z2j, var_z2j)
        return {
            "mu_x": mu_x,
            "var_x": var_x,
            "mu_z1": mu_z1,
            "var_z1": var_z1,
            "mu_h": mu_h,
            "var_h": var_h,
            "mu_z2j": mu_z2j,
            "var_z2j": var_z2j,
            "mu_z2": mu_z2,
            "var_z2": var_z2,
        }

    def predict(self, x):
        """Predictive mean and variance of the output (including β²)."""
        cache = self._forward_cache(x)
        return cache["mu_z2"], cache["var_z2"] + self.beta2

    def one_sweep(self, x, y):
        """One forward + backward sweep; multiply product-backward into weights."""
        cache = self._forward_cache(x)
        mu_y = jnp.asarray(y, dtype=jnp.float64)
        var_y = self.beta2

        # Output sum: cavity to each product term z2_j.
        mu_cav2, var_cav2 = sum_cavity(
            mu_y,
            var_y,
            cache["mu_z2"],
            cache["var_z2"],
            cache["mu_z2j"],
            cache["var_z2j"],
        )
        tau_cav2, rho_cav2 = moments_to_natural(mu_cav2, var_cav2)
        tau_h, rho_h = moments_to_natural(cache["mu_h"], cache["var_h"])
        tau_w2, rho_w2 = moments_to_natural(self.mu_w2, self.var_w2)

        # Product backward to W2 (Y = hidden activation, Z = cavity).
        tau_msg_w2, rho_msg_w2 = product_backward(tau_h, rho_h, tau_cav2, rho_cav2)
        # Product backward to h uses incoming W2, not the updated belief.
        tau_h_bwd, rho_h_bwd = product_backward(tau_w2, rho_w2, tau_cav2, rho_cav2)
        tau_w2, rho_w2 = multiply_natural(tau_w2, rho_w2, tau_msg_w2, rho_msg_w2)
        self.mu_w2, self.var_w2 = natural_to_moments(tau_w2, rho_w2)
        mu_h_bwd, var_h_bwd = natural_to_moments(tau_h_bwd, rho_h_bwd)

        # Leaky-ReLU backward to z1 (Prop. 3.5): uses only the message at h.
        mu_z1_bwd, var_z1_bwd = relu_backward(mu_h_bwd, var_h_bwd, self.alpha)

        # Hidden sum is a single product (d_in = 1): cavity = backward on z1.
        tau_z1_bwd, rho_z1_bwd = moments_to_natural(mu_z1_bwd, var_z1_bwd)
        tau_x, rho_x = moments_to_natural(cache["mu_x"], cache["var_x"])
        tau_w1, rho_w1 = moments_to_natural(self.mu_w1, self.var_w1)
        tau_msg_w1, rho_msg_w1 = product_backward(
            tau_x, rho_x, tau_z1_bwd, rho_z1_bwd
        )
        tau_w1, rho_w1 = multiply_natural(tau_w1, rho_w1, tau_msg_w1, rho_msg_w1)
        self.mu_w1, self.var_w1 = natural_to_moments(tau_w1, rho_w1)
        return self
