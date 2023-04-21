#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A collection optional functions,
which can be used for better functionality.

"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import quantity_support
from os import path, listdir
from glob import glob
from os.path import getsize
from prettytable import PrettyTable
from importlib_resources import files


data_dir = path.join(path.dirname(__file__), "data")


def merge_fits(
    aeff_fits=path.join(data_dir, "aeff.fits"),
    psf_fits=path.join(data_dir, "psf.fits"),
    edisp_fits=path.join(data_dir, "edisp.fits"),
    bkg_fits=path.join(data_dir, "bkg_nu.fits"),
    output_path=data_dir,
    output_file="all_in_one.fits",
):
    r"""
    Merge separated .fits files into one, which can be used in gammapy

    Parameters
    ----------
    aeff_fits : str
        path to Aeff .fits file
    psf_fits : str
        path  to PSF .fits file
    edisp_fits : str
        path to Edisp .fits file
    bkg_fits : str
        path to Background .fits file
    output_path : str
        path for the merged IRF file
    output_file : str
        name of the merged .fits file in data foledr of the package.
        .fits should be included in the title.

    Returns
    -------
    None
    """
    hdu_list = []
    hdu_list.append(fits.PrimaryHDU())

    file_aeff = fits.open(aeff_fits)
    hdu_list.append(file_aeff[1])
    hdu_list[1].name = "EFFECTIVE AREA"

    file_psf = fits.open(psf_fits)
    hdu_list.append(file_psf[1])
    hdu_list[2].name = "POINT SPREAD FUNCTION"

    file_edisp = fits.open(edisp_fits)
    hdu_list.append(file_edisp[1])
    hdu_list[3].name = "ENERGY DISPERSION"

    file_bkg = fits.open(bkg_fits)
    hdu_list.append(file_bkg[1])
    hdu_list[4].name = "BACKGROUND"

    new_fits_file = fits.HDUList(hdu_list)
    new_fits_file.writeto(path.join(output_path, output_file), overwrite=True)

    file_aeff.close()
    file_psf.close()
    file_edisp.close()
    file_bkg.close()

    print(f"combined IRF file {output_file} is merged successfully!")

    return None


def list_data(print_tab=False):
    r"""
    Return dictionary of .fits files with names and pathes in the data folder

    Parameters
    ----------
    print_tab : bool, default False
        print in terminal a table with content of data folder

    Returns
    -------
    dict
        dictionary of files
    """
    tab = PrettyTable(["File Path", "Size, KB"], align="l")
    data_path = path.join(data_dir, "*.fits")
    info = {}

    clean_list = [i for i in listdir(data_dir) if ".fits" in i]
    for file, i in zip(glob(data_path, recursive=True), clean_list):
        if ".fits" in i:
            tab.add_row([file, round(getsize(filename=file) / float(1 << 10), 2)])
            info.update({i: file})
    # show something
    if print_tab:
        print(tab)

    return info


class DrawAeff:
    """Class is responsible for production of Aeff plots."""

    np.seterr(divide="ignore")

    def __init__(self, aeff_path=path.join(data_dir, "aeff_nocuts.fits")):
        self.aeff_path = aeff_path
        with fits.open(self.aeff_path) as hdul:
            self.data = hdul[1].data
            self.head = hdul[1].header

        self.energy_center = np.log10(
            (self.data["ENERG_HI"][0] + self.data["ENERG_LO"][0]) / 2.0
        )
        self.zenith = (
            np.cos(self.data["THETA_HI"][0]) + np.cos(self.data["THETA_LO"][0])
        ) / 2.0

    def plot_energy_dependence(
        self,
        ax=None,
        zenith_index=None,
        **kwargs,
    ):
        """
        Plot effective area versus energy for a given zenith angle.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`, optional
            Axis
        zenith_index : List
            list of items in zenith axes
        kwargs : dict
            Forwarded tp plt.plot()
        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            Axis
        """
        ax = plt.gca() if ax is None else ax

        if zenith_index is None:
            zenith_index = np.linspace(0, len(self.zenith) - 1, 4, dtype=int)

        for zen in zenith_index:
            area = np.nan_to_num(np.log10(self.data["EFFAREA"][0][zen][:]), neginf=-3)
            cos_zen = "{:.2f}".format(self.zenith[zen])
            label = kwargs.pop("label", r"$ \cos(\theta)$=" + cos_zen)
            with quantity_support():
                ax.plot(self.energy_center, area, label=label, **kwargs)

        ax.set_xlabel(f"log(E) [{self.head['TUNIT1']}]")
        ax.set_ylabel(f"log(Effective Area) [{self.head['TUNIT5']}]")
        ax.legend()
        return ax

    def plot_zenith_dependence(
        self,
        ax=None,
        energy_index=None,
        **kwargs,
    ):
        """
        Plot effective area versus cosine of zenith angle for a given energy.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`, optional
            Axis
        energy_index : `~astropy.units.Quantity`
            Energy
        **kwargs : dict
            Keyword argument passed to `~matplotlib.pyplot.plot`
        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            Axis
        """
        ax = plt.gca() if ax is None else ax

        if energy_index is None:
            energy_index = np.linspace(0, len(self.energy_center) - 1, 4, dtype=int)

        for en in energy_index:
            data_T = np.transpose(self.data["EFFAREA"][0])
            area = np.nan_to_num(np.log10(data_T[en][:]), neginf=-3)
            log_e = "{:.2f}".format(self.energy_center[en])
            label = kwargs.pop("label", f"log(E)={log_e}")
            with quantity_support():
                ax.plot(self.zenith, area, label=label, **kwargs)

        ax.set_xlabel(r"$ \cos(\theta)$")
        ax.set_ylabel(f"log(Effective Area) [{self.head['TUNIT5']}]")
        ax.legend()

        # if energy is None:
        #     energy_axis = self.axes["energy_true"]
        #     e_min, e_max = energy_axis.center[[0, -1]]
        #     energy = np.geomspace(e_min, e_max, 4)

        # offset_axis = self.axes["offset"]

        # for ee in energy:
        #     area = self.evaluate(offset=offset_axis.center, energy_true=ee)
        #     area /= np.nanmax(area)
        #     if np.isnan(area).all():
        #         continue
        #     label = f"energy = {ee:.1f}"
        #     with quantity_support():
        #         ax.plot(offset_axis.center, area, label=label, **kwargs)

        # offset_axis.format_plot_xaxis(ax=ax)
        # ax.set_ylim(0, 1.1)
        # ax.set_ylabel("Relative Effective Area")
        # ax.legend(loc="best")
        return ax

    def plot_aeff(self, ax=None, add_cbar=True, **kwargs):
        """
        Plot effective area image.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`, optional
            Axis
        add_cbar : bool, default True
            add color bar to plot
        **kwargs : dict
            Keyword argument passed to `~matplotlib.pyplot.plot`

        """

        ax = plt.gca() if ax is None else ax
        Y, X = np.meshgrid(self.energy_center, self.zenith)
        Z = np.nan_to_num(np.log10(self.data["EFFAREA"][0]), neginf=-3)

        vmin, vmax = np.nanmin(Z), np.nanmax(Z)

        kwargs.setdefault("cmap", "RdPu")
        kwargs.setdefault("edgecolors", "face")
        kwargs.setdefault("shading", "auto")
        kwargs.setdefault("vmin", vmin)
        kwargs.setdefault("vmax", vmax)

        with quantity_support():
            caxes = ax.pcolormesh(X, Y, Z, **kwargs)

        ax.axes.set_xlabel(r"$ \cos(\theta)$")
        ax.axes.set_ylabel(f"log(E) [{self.head['TUNIT1']}]")

        if add_cbar:
            label = f"log(Effective Area) [{self.head['TUNIT5']}]"
            ax.figure.colorbar(caxes, ax=ax, label=label)

        return ax

    def peek(self, figsize=(15, 4)):
        """
        Quick-look summary plots for Aeff.

        Parameters
        ----------
        figsize : tuple
            Size of the figure.
        """
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=figsize)
        self.plot_energy_dependence(ax=axes[0])
        self.plot_zenith_dependence(ax=axes[1])
        self.plot_aeff(ax=axes[2])
        plt.tight_layout()
