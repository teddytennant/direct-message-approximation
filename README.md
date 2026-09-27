Direct Message Approximation (DMA) from [arXiv:2609.29466](https://arxiv.org/abs/2609.29466) (Herbrich et al.): JAX closed-form Gaussian messages for the product factor δ(z − xy) and leaky-ReLU factor δ(y − ReLU_α(x)), plus a one-sweep 1-hidden-layer scalar BNN.

```
pip install -e ".[tests]"
pytest
python train.py
```

What does not match the paper: this is not the authors' Julia reference; no full ICLR 1D regression figure with the 7-d feature map; no EP mini-batch cavity schedule; CPU only. Variances are clipped to a tiny positive floor. Theorem 3.3's O(1/r²) bound is not checked numerically (that needs the truncated-ratio reference of Appendix B.5).
