# Notebooks

The reference models are trained by the Kaggle kernels generated from
[`training/`](../training) (`python training/kaggle.py push <benchmark>`), which
reuse the server's own split code. Fork those kernels to start a submission.

`legacy/` holds the v1 Colab notebooks. They produce **split_v1** ids
(filename-based, see ASSESSMENT.md §1), which the v2 benchmarks no longer accept.
