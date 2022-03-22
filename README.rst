KM3NeT instrument response functions
====================================

.. image:: https://git.km3net.de/km3py/km3irf/badges/master/pipeline.svg
    :target: https://git.km3net.de/km3py/km3irf/pipelines

.. image:: https://git.km3net.de/km3py/km3irf/badges/master/coverage.svg
    :target: https://km3py.pages.km3net.de/km3irf/coverage

.. image:: https://git.km3net.de/examples/km3badges/-/raw/master/docs-latest-brightgreen.svg
    :target: https://km3py.pages.km3net.de/km3irf


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
