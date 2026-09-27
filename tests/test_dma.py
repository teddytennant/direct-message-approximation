"""Lock Propositions 3.1, 3.2, 3.4, 3.5 of arXiv:2609.29466."""

from __future__ import annotations

import inspect
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dma import TinyBNN, product_backward, product_forward, relu_backward, relu_forward
from dma.gaussian import moments_to_natural, natural_to_moments
from dma.product import product_backward_moments

jax.config.update("jax_enable_x64", True)

RTOL = 1e-10
ATOL = 1e-10


def _np(x):
    return np.asarray(x, dtype=np.float64)


def assert_close(a, b, rtol=RTOL, atol=ATOL):
    np.testing.assert_allclose(_np(a), _np(b), rtol=rtol, atol=atol)


def phi(u):
    return math.exp(-0.5 * u * u) / math.sqrt(2.0 * math.pi)


def Phi(u):
    return 0.5 * (1.0 + math.erf(u / math.sqrt(2.0)))


# --- Proposition 3.1 / eq. (6) ------------------------------------------------


@pytest.mark.parametrize(
    "mu_x, var_x, mu_y, var_y, mz, sz2",
    [
        (1.0, 1.0, 2.0, 1.0, 2.0, 6.0),
        (0.0, 1.0, 0.0, 1.0, 0.0, 1.0),
        (3.0, 0.0, 4.0, 0.0, 12.0, 0.0),
        (0.0, 4.0, 5.0, 0.25, 0.0, 101.0),
        (-1.0, 4.0, 3.0, 0.25, -3.0, 37.25),
        (2.0, 0.0, 5.0, 1.0, 10.0, 4.0),
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    ],
)
def test_product_forward_eq6(mu_x, var_x, mu_y, var_y, mz, sz2):
    m, s2 = product_forward(mu_x, var_x, mu_y, var_y)
    assert_close(m, mz)
    assert_close(s2, sz2)


def test_product_forward_dirac_consistency():
    mz, sz2 = product_forward(2.0, 1e-24, 3.0, 1e-24)
    assert_close(mz, 6.0)
    assert abs(float(sz2)) < 1e-20
    mz0, sz20 = product_forward(1.5, 0.0, -4.0, 0.0)
    assert_close(mz0, -6.0)
    assert_close(sz20, 0.0)


# --- Proposition 3.2 / eq. (7) ------------------------------------------------


def test_product_backward_eq7_hand_computed():
    # τy=2, ρy=1, τz=3, ρz=1
    # r = 2^4 * 1 / (1 * (4 + 1)) = 16/5
    # D = 9*1 + 4*1 + 1*1 = 14
    # τ̂x = 2 * (16/5) * 3 / 14 = 48/35
    # ρ̂x = (16/5)^2 / 14 = 128/175
    tau_hat, rho_hat = product_backward(2.0, 1.0, 3.0, 1.0)
    assert_close(tau_hat, 48.0 / 35.0)
    assert_close(rho_hat, 128.0 / 175.0)


def test_product_backward_eq7_formula():
    tau_y, rho_y, tau_z, rho_z = -1.5, 0.8, 2.0, 1.25
    r = (tau_y ** 4) * rho_z / (rho_y * (tau_y ** 2 + rho_y))
    D = (tau_z ** 2) * rho_y + (tau_y ** 2) * rho_z + rho_y * rho_z
    tau_hat, rho_hat = product_backward(tau_y, rho_y, tau_z, rho_z)
    assert_close(tau_hat, (tau_y * r * tau_z) / D)
    assert_close(rho_hat, (r ** 2) / D)


def test_product_backward_dirac_y():
    mu_y, var_y = 2.0, 1e-10
    mu_z, var_z = 6.0, 0.25
    mu_x, var_x = product_backward_moments(mu_y, var_y, mu_z, var_z)
    assert_close(mu_x, mu_z / mu_y, rtol=1e-6, atol=1e-6)
    assert_close(var_x, var_z / (mu_y ** 2), rtol=1e-6, atol=1e-6)


# --- Proposition 3.4 / eq. (10) -----------------------------------------------


def _relu_forward_paper(mu_x, var_x, alpha):
    sigma_x = math.sqrt(var_x)
    u = mu_x / sigma_x
    P, ph = Phi(u), phi(u)
    A = alpha + (1.0 - alpha) * P
    B = alpha ** 2 + (1.0 - alpha ** 2) * P
    mY = mu_x * A + (1.0 - alpha) * sigma_x * ph
    s2Y = (mu_x ** 2 + var_x) * B + (1.0 - alpha ** 2) * mu_x * sigma_x * ph - mY ** 2
    return mY, s2Y


@pytest.mark.parametrize(
    "mu_x, var_x, alpha",
    [
        (1.0, 1.0, 0.2),
        (0.0, 1.0, 0.1),
        (-0.5, 0.25, 0.3),
        (2.0, 0.04, 0.01),
        (0.3, 2.0, 0.5),
    ],
)
def test_relu_forward_eq10(mu_x, var_x, alpha):
    m, s2 = relu_forward(mu_x, var_x, alpha)
    m_ref, s2_ref = _relu_forward_paper(mu_x, var_x, alpha)
    assert_close(m, m_ref)
    assert_close(s2, s2_ref)


def test_relu_forward_identity_alpha_1():
    for mu_x, var_x in [(0.0, 1.0), (1.3, 0.4), (-2.0, 2.5), (0.0, 0.01)]:
        m, s2 = relu_forward(mu_x, var_x, 1.0)
        assert_close(m, mu_x)
        assert_close(s2, var_x)


# --- Proposition 3.5 / eqs. (11)–(12) -----------------------------------------


def _relu_backward_paper(mu_y, var_y, alpha):
    sigma_y = math.sqrt(var_y)
    v = mu_y / sigma_y
    P, ph = Phi(v), phi(v)
    Q = 1.0 - P
    C = alpha * P + Q
    mX = (mu_y * (alpha ** 2 * P + Q) + (alpha ** 2 - 1.0) * sigma_y * ph) / (alpha * C)
    s2X = (
        (mu_y ** 2 + var_y) * (alpha ** 3 * P + Q)
        + (alpha ** 3 - 1.0) * mu_y * sigma_y * ph
    ) / (alpha ** 2 * C) - mX ** 2
    return mX, s2X


@pytest.mark.parametrize(
    "mu_y, var_y, alpha",
    [
        (1.0, 1.0, 0.2),
        (0.0, 1.0, 0.1),
        (-0.4, 0.81, 0.25),
        (1.5, 0.09, 0.05),
    ],
)
def test_relu_backward_eq11_12(mu_y, var_y, alpha):
    m, s2 = relu_backward(mu_y, var_y, alpha)
    m_ref, s2_ref = _relu_backward_paper(mu_y, var_y, alpha)
    assert_close(m, m_ref)
    assert_close(s2, s2_ref)


def test_relu_backward_identity_alpha_1():
    for mu_y, var_y in [(0.0, 1.0), (2.0, 0.5), (-1.0, 3.0)]:
        m, s2 = relu_backward(mu_y, var_y, 1.0)
        assert_close(m, mu_y)
        assert_close(s2, var_y)


# --- Positive precision -------------------------------------------------------


def test_messages_positive_precision():
    pairs = [
        (0.5, 0.4, 1.2, 0.9),
        (-1.0, 0.2, 0.3, 1.5),
        (2.0, 3.0, -0.7, 0.8),
        (0.1, 0.1, 0.1, 0.1),
    ]
    for mu_x, var_x, mu_y, var_y in pairs:
        _, sz2 = product_forward(mu_x, var_x, mu_y, var_y)
        assert np.isfinite(_np(sz2)) and float(sz2) > 0.0
        tau_y, rho_y = moments_to_natural(mu_y, var_y)
        tau_z, rho_z = moments_to_natural(0.4, 0.7)
        tau_hat, rho_hat = product_backward(tau_y, rho_y, tau_z, rho_z)
        assert np.isfinite(_np(tau_hat)) and np.isfinite(_np(rho_hat))
        assert float(rho_hat) > 0.0
        mu_hat, var_hat = natural_to_moments(tau_hat, rho_hat)
        assert float(var_hat) > 0.0 and np.isfinite(_np(mu_hat))

    for alpha in (0.05, 0.2, 0.9, 1.0):
        for mu, var in [(0.0, 1.0), (1.5, 0.3), (-2.0, 2.0), (0.2, 0.05)]:
            mY, s2Y = relu_forward(mu, var, alpha)
            assert np.isfinite(_np(mY)) and np.isfinite(_np(s2Y)) and float(s2Y) > 0.0
            mX, s2X = relu_backward(mu, var, alpha)
            assert np.isfinite(_np(mX)) and np.isfinite(_np(s2X)) and float(s2X) > 0.0


# --- DMA definition: no same-edge cavity --------------------------------------


def test_dma_outgoing_independent_of_same_edge_incoming():
    """Outgoing factor-to-variable messages ignore the incoming on that edge."""
    sig_pf = inspect.signature(product_forward)
    sig_pb = inspect.signature(product_backward)
    sig_rf = inspect.signature(relu_forward)
    sig_rb = inspect.signature(relu_backward)
    assert "mu_z" not in sig_pf.parameters and "tau_z_in" not in sig_pf.parameters
    assert "tau_x" not in sig_pb.parameters and "rho_x" not in sig_pb.parameters
    assert "mu_y" not in sig_rf.parameters
    assert "mu_x" not in sig_rb.parameters

    def product_to_z(msg_x, msg_y, msg_z_incoming):
        del msg_z_incoming
        return product_forward(*msg_x, *msg_y)

    def product_to_x(msg_y, msg_z, msg_x_incoming):
        del msg_x_incoming
        return product_backward(*msg_y, *msg_z)

    def relu_to_y(msg_x, alpha, msg_y_incoming):
        del msg_y_incoming
        return relu_forward(*msg_x, alpha)

    def relu_to_x(msg_y, alpha, msg_x_incoming):
        del msg_x_incoming
        return relu_backward(*msg_y, alpha)

    x_a, x_b = (1.0, 0.5), (-3.0, 4.0)
    y_a, y_b = (0.7, 1.1), (2.0, 0.3)
    z_a, z_b = (0.2, 0.9), (-1.0, 2.5)
    assert_close(product_to_z(x_a, y_a, z_a), product_to_z(x_a, y_a, z_b))
    ty_a, ry_a = moments_to_natural(*y_a)
    tz_a, rz_a = moments_to_natural(*z_a)
    assert_close(
        product_to_x((ty_a, ry_a), (tz_a, rz_a), x_a),
        product_to_x((ty_a, ry_a), (tz_a, rz_a), x_b),
    )
    assert_close(relu_to_y(x_a, 0.2, y_a), relu_to_y(x_a, 0.2, y_b))
    assert_close(relu_to_x(y_a, 0.2, x_a), relu_to_x(y_a, 0.2, x_b))


# --- Tiny BNN one-sweep -------------------------------------------------------


def test_bnn_one_sweep_finite_positive_variances():
    bnn = TinyBNN(d_hidden=2, alpha=0.2, beta=0.3, sigma0=1.0, key=jax.random.PRNGKey(1))
    bnn.one_sweep(1.0, 0.5)
    for mu, var in (
        (bnn.mu_w1, bnn.var_w1),
        (bnn.mu_w2, bnn.var_w2),
    ):
        assert np.all(np.isfinite(_np(mu)))
        assert np.all(np.isfinite(_np(var)))
        assert np.all(_np(var) > 0.0)
    pm, pv = bnn.predict(1.0)
    assert np.isfinite(_np(pm)) and np.isfinite(_np(pv)) and float(pv) > 0.0
