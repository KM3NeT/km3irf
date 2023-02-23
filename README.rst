.. image:: https://git.km3net.de/km3py/km3irf/badges/main/pipeline.svg
    :target: https://git.km3net.de/km3py/km3irf/pipelines

.. image:: https://git.km3net.de/km3py/km3irf/badges/main/coverage.svg
    :target: https://km3py.pages.km3net.de/km3irf/coverage

.. image:: https://git.km3net.de/examples/km3badges/-/raw/master/docs-latest-brightgreen.svg
    :target: https://km3py.pages.km3net.de/km3irf


KM3NeT instrument response function
====================================

This project provides a versatile tool that can be used to quickly analyze the sensitivity of the **KM3NeT** detector for various source models.
Currently it considers only point-like sources. The main feature of the tool is deep targeting to ``gammapy`` software.
For further analysis in ``gammapy``, ``km3irf`` provides next modules:

* Event list
* Instrument response function (IRF)
* Data set

Installation
~~~~~~~~~~~~

It is recommended to first create an isolated virtualenvironment to not interfere
with other Python projects::

  git clone https://git.km3net.de/km3py/km3irf
  cd km3irf
  python3 -m venv venv
  . venv/bin/activate

Install directly from the Git server via ``pip`` (no cloneing needed)::

  pip install git+https://git.km3net.de/km3py/km3irf

Or clone the repository and run::

  make install

To install all the development dependencies, in case you want to contribute or
run the test suite::

  make install-dev
  make test


---

*Created with ``cookiecutter https://git.km3net.de/templates/python-project``*
