.. HadamardLangevin documentation master file, created by
   sphinx-quickstart on Tue Nov  5 15:20:57 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

HadamardLangevin documentation
===============================



.. toctree::
   :hidden:
   :maxdepth: 1
   :glob:
   :caption: Getting started

   vignettes/*

.. toctree::
   :hidden:
   :maxdepth: 3
   :caption: API

   HadamardLangevin

This is a Python package for sampling from densities of the form  :math:`\pi(x)\propto \exp(-\beta(\lambda \|x\|_1 + G(x)))`, where :math:`G:\mathbb{R}^n\to \mathbb{R}` is a differentiable function, and :math:`\lambda,\beta>0`. The goal is to demonstrate the use of the Hadamard product.


HadamardLangevin package description
-------------------------------------

- **samplers.py**: implementation of proximal and Hadamard Langevin sampling methods.
- **utils.py**: various useful functions for inverse problems, such as wrappers for the wavelet transform.



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

