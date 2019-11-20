.. clar documentation master file, created by
   sphinx-quickstart on Tue Nov  5 15:39:59 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Installation
------------
First clone the repository available at https://github.com/qb3/clar::

    $ git clone https://github.com/qb3/clar.git
    $ cd clar/


We recommend to use the `Anaconda Python distribution <https://www.continuum.io/downloads>`_,
and create a conda environment with::

    $ conda env create --file environment.yml

Then, you can compile the Cython code and install the package with::

    $ source activate clar-env
    $ pip install --no-deps -e .

To check if everything worked fine, you can do::

    $ source activate clar-env
    $ python -c 'import clar'

and it should not give any error message.

From a Python shell you can just do::

    >>> import clar

If you don't want to use Anaconda, the list of packages you need to install is in the `environment.yml` file.

Cite
----

If you use this code, please cite:

.. code-block:: None

    @article{bertrand2019handling,
    title={Handling correlated and repeated measurements with the smoothed multivariate square-root {L}asso},
    author={Bertrand, Quentin and Massias, Mathurin and Gramfort, Alexandre and Salmon, Joseph},
    year={2019},
    journal={NeurIPS}
    }



ArXiv link: https://arxiv.org/abs/1902.02509



API
---

.. toctree::
    :maxdepth: 1

    api.rst