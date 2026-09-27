"""1D regression smoke: one-sweep DMA BNN on a handful of points."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import numpy as np

from dma import TinyBNN

jax.config.update("jax_enable_x64", True)


def main():
    xs = np.array([-1.5, -0.8, -0.2, 0.4, 1.0, 1.6], dtype=np.float64)
    ys = 0.4 * xs + 0.15 * xs ** 3
    bnn = TinyBNN(
        d_hidden=2,
        alpha=0.2,
        beta=0.25,
        sigma0=1.0,
        key=jax.random.PRNGKey(0),
    )
    for x, y in zip(xs, ys):
        bnn.one_sweep(float(x), float(y))

    grid = np.linspace(-2.0, 2.0, 41, dtype=np.float64)
    means, vars_ = [], []
    for x in grid:
        m, v = bnn.predict(float(x))
        means.append(float(np.asarray(m)))
        vars_.append(float(np.asarray(v)))

    out = {
        "x_train": xs.tolist(),
        "y_train": ys.tolist(),
        "x_grid": grid.tolist(),
        "pred_mean": means,
        "pred_var": vars_,
        "weight_means": {
            "w1": np.asarray(bnn.mu_w1).tolist(),
            "w2": np.asarray(bnn.mu_w2).tolist(),
        },
        "weight_vars": {
            "w1": np.asarray(bnn.var_w1).tolist(),
            "w2": np.asarray(bnn.var_w2).tolist(),
        },
    }
    path = Path(__file__).resolve().parent / "results.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"wrote {path}")
    print(
        "predictive at 0:",
        f"mean={out['pred_mean'][20]:.4f}",
        f"var={out['pred_var'][20]:.4f}",
    )


if __name__ == "__main__":
    main()
