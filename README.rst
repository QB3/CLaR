CLaR
=====

|image0|

This package implements the CLaR alorithm, a fast algorithm to handle sparse linear regression with heteroscedastic noise, see:
Bertrand, Q., Massias, M., Gramfort, A., & Salmon, J. (2019). Concomitant Lasso with Repetitions (CLaR): beyond averaging multiple realizations of heteroscedastic noise. arXiv preprint arXiv:1902.02509.

It also implements a variation of the mrce algorithm with a L21 penalization on the regression coefficient Beta, and which take in account the repetitions see
Rothman, A. J., Levina, E., & Zhu, J. (2010). Sparse multivariate regression with covariance estimation. Journal of Computational and Graphical Statistics, 19(4), 947-962.


To be able to run the code you first need to run, in this folder:
```pip install -e .```

After that,
```python -c "import clar"```
should run smoothly, and you can execute the `example.py` file.

ArXiv link: https://export.arxiv.org/abs/1902.02509

.. |image0| image:: https://travis-ci.org/QB3/CLaR.svg?branch=master
   :target: https://travis-ci.org/QB3/CLaR/
