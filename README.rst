CLaR
=====

|image0| |image1|

Fast algorithm to solve high dimensional linear regression with correlated noise and multiple measurements


Documentation
=============

Please visit https://qb3.github.io/CLaR/ for the latest version
of the documentation.


Install the released version
============================

To setup a fully functional environment we recommend you download this
`conda
environment <https://raw.githubusercontent.com/QB3/CLaR/master/environment.yml>`__
and install it with:

::

    conda env create --file environment.yml

Install the development version
===============================

From a console or terminal clone the repository and install CLaR:

::

    git clone https://github.com/QB3/CLaR.git
    conda env create --file environment.yml
    source activate clar-env
    pip install --no-deps -e .

Demos & Examples
================

You can find examples `here <https://qb3.github.io/CLaR/auto_examples/index.html>`__

Dependencies
============

All dependencies are in ``./environment.yml``

Cite
====

If you use this code, please cite:

::

    @article{bertrand2019handling,
    title={Handling correlated and repeated measurements with the smoothed multivariate square-root {L}asso},
    author={Bertrand, Quentin and Massias, Mathurin and Gramfort, Alexandre and Salmon, Joseph},
    year={2019},
    journal={NeurIPS}
    }

::

ArXiv link: https://export.arxiv.org/abs/1902.02509

.. |image0| image:: https://travis-ci.org/QB3/CLaR.svg?branch=master
   :target: https://travis-ci.org/QB3/CLaR/

.. |image1| image:: https://codecov.io/gh/QB3/CLaR/branch/master/graphs/badge.svg?branch=master
   :target: https://codecov.io/gh/QB3/CLaR
