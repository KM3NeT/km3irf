"""
Example № 1
===========

This example produces *Effective Area* (Aeff), 
*Point Spread Function* (PSF), *Energy Dispersion* (Edisp) 
files in .fits format from original KM3NeT 
simulation dst.root file. And finally merge them into one common
.fits file.
"""

from km3net_testdata import data_path
from km3irf import build_irf
from astropy.io import fits
from km3irf.utils import merge_fits

import warnings

warnings.simplefilter("ignore")

# %%
# Define a path to 10 events data file from ``km3net_testdata``

path_to_data = data_path("dst/mcv5.1.km3_numuCC.ALL.dst.bdt.10events.root")

# %%
# Effective Area
# --------------
# Create ``DataContainer`` object from test ``.dst`` file

test_irf = build_irf.DataContainer(path_to_data)

# %%
# Check, how many events in the file

test_irf.df.shape[0]


# %%
# Apply default cuts, but they also can be changed by user

test_irf.apply_cuts()

# %%
# Check, how many events survived after cuts:

test_irf.df.shape[0]

# %%
# Apply re-weighting procedure for calculation new weights, which correspond
# to a new spectral index. Weights are not needed for *Effective Area*.

weighted_dict = test_irf.weight_calc(tag="nu")
weighted_dict.values()

# %%
# To create ``.fits`` for *Effective area*, we need to pass a pandas data frame as input.
# Default name is ``aeff.fits``.

test_irf.build_aeff()

# %%
# Point Spread Function
# ---------------------
# In this part we create ``.fits`` for *Point Spread Function*.
# Default name is ``psf.fits``.

test_irf.build_psf()

# %%
# Energy dispertion
# -----------------
# In this part we create ``.fits`` for *Energy dispertion*
# Default name is ``edisp.fits``.

test_irf.build_edisp(norm=True, smooth=False, smooth_norm=False)

# %%
# Merge into one file
# -------------------
# All ``.fits`` files can be merged in one ``.fits`` file for convenience.
# Missing background file can be taken from data folder of the package.
# Later the creation of background ``.fits`` file will be added.

merge_fits(
    aeff_fits="aeff.fits",
    psf_fits="psf.fits",
    edisp_fits="edisp.fits",
    output_path=".",
    output_file="new_IRF.fits",
)

# %%
# Check content with gammapy
# --------------------------
# | Here is an example how to test created `.fits` files with `gammapy <https://gammapy.org/>`_.
# | Firstly it requires to install gammapy.
# | ``!pip install gammapy``
# | ``from gammapy.irf import EffectiveAreaTable2D, PSF3D, EnergyDispersion2D``
# | ``aeff = EffectiveAreaTable2D.read("aeff.fits", hdu="EFFECTIVE AREA")``
# | ``print(aeff)``
# | Similar procedure shouldbe repeated with other ``.fits`` files.
