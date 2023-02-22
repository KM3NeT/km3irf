"""
Example № 1
===========

This example produces Effective Area (Aeff), 
Point Spread Function (PSF), Energy Dispersion (Edisp) 
files in .fits format from original KM3NeT 
simulation dst.root file. And finally merge them into one common
.fits file.
"""

from km3irf import build_irf
from astropy.io import fits

# %%
# Define a path to your local `dst.root` file:

data_path = "/run/media/msmirnov/DATA2/data_files/IRF_data_create/mcv5.1.km3_numuCC.ALL.dst.bdt.root"

# %%
# Effective Area
# --------------
# Create BuildAeff object:

test_irf = build_irf.DataContainer(data_path)
