"""
Example № 2
===========

This example demonstrates the drawing features of the package for separated IRFs files.
"""
# %%
# Currently Aeff, Edisp and PSF have drawing functionality in the same manner as ``gammapy``.
# As data it uses default IRFs files from ``/data`` directory.

from km3irf import utils, psf_3d
from os import path
import matplotlib.pyplot as plt

# %%
# Effective area
# --------------
# It is possible to draw Aeff as a function of energy, zenith angle and finally two dimensional plot with both dependencies.
# The summary plots can be created using ``peek()`` function.

# create an instance of the DrawAeff class
path_to_aeff = path.join(utils.data_dir, "aeff.fits")
aeff = utils.DrawAeff(path_to_aeff)

# %%
# Draw the Aeff as a function of energy.

aeff.plot_energy_dependence()
plt.show()

# %%
# Draw the Aeff as a function of zenith angle.

aeff.plot_zenith_dependence()
plt.show()

# %%
# Draw the Aeff as a function of energy and zenith angle.

aeff.plot_aeff()
plt.show()

# %%
# Summary plots.

aeff.peek()
plt.show()

# %%
# Energy dispertion
# -----------------
# It creates next plots:
#
# * Edisp two dimensional PDF as a function of true energy and migration for a given zenith angle
# * Edisp for given zenith angle and true energy as a function of migration

# create an instance of the DrawEdisp class
edisp = utils.DrawEdisp()

# %%
# Draw the two dimensional PDF as a function of true energy and migration for a given zenith angle.
edisp.plot_bias()
plt.show()

# %%
# Draw the Edisp for given zenith angle and true energy as a function of migration.

edisp.plot_migration()
plt.show()

# %%
# Summary plots.

edisp.peek()
plt.show()

# %%
# Point spread function
# ---------------------
# To provide better functionality, two modules ``psf_3d.py`` and ``psf_table.py`` were copied from ``gammapy``.
# And appropiate changes were made to adapt them for the purposes of the package.
# Currently, there is no class ``DrawPSF`` in ``utils.py``.

# create an instance of the PSF3D class using read method
psf = psf_3d.PSF3D.read(path.join(utils.data_dir, "psf.fits"))

# %%
# Draw PSF as a function of rad for fixed zenith theta angle and energy.

psf.plot_psf_vs_rad()
plt.show()

# %%
# Draw containment radius for fixed fraction and as a function of energy and zenith angle.

psf.plot_containment()
plt.show()

# %%
# Draw containment fraction as a function of energy for fixed zenith theta angle.

psf.plot_containment_vs_energy()
plt.show()

# %%
# Summary plots.

psf.peek()
plt.show()
