#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A collection optional functions,
which can be used for better functionality.

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from astropy.io import fits
from astropy.visualization import quantity_support
import astropy.units as u
from os import path, listdir
from glob import glob
from os.path import getsize
from prettytable import PrettyTable
from importlib_resources import files
from .interpolation import ScaledRegularGridInterpolator
import scipy as sp


data_dir = path.join(path.dirname(__file__), "data")


def merge_fits(
    aeff_fits=path.join(data_dir, "aeff.fits"),
    psf_fits=path.join(data_dir, "psf.fits"),
    edisp_fits=path.join(data_dir, "edisp.fits"),
    bkg_fits=path.join(data_dir, "bkg_nu.fits"),
    output_path=data_dir,
    output_file="all_in_one.fits",
):
    r"""Merge separated .fits files into one, which can be used in gammapy

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
    """
    Return dictionary of .fits files with names and paths in the data folder

    Parameters
    ----------
    print_tab : bool, default False
        print in terminal a table with the content of the data folder

    Returns
    -------
    dict
        dictionary of files
    """
    info = {}

    clean_list = [i for i in listdir(data_dir) if ".fits" in i]
    for file, i in zip(glob(path.join(data_dir, "*.fits"), recursive=True), clean_list):
        info[i] = file

    if print_tab:
        tab = PrettyTable(["File Path", "Size, KB"], align="l")
        for i, file in info.items():
            tab.add_row([file, round(getsize(filename=file) / float(1 << 10), 2)])
        print(tab)
        return None

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
        energy_index : List
            list of items in energy axes
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
        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            Axis

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
        """Quick-look summary plots for Aeff.

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


class DrawEdisp:
    """Class is responsible for production of Edisp plots."""

    np.seterr(divide="ignore")

    def __init__(self, edisp_path=path.join(data_dir, "edisp.fits")):
        self.edisp_path = edisp_path
        with fits.open(self.edisp_path) as hdul:
            self.data = hdul[1].data
            self.head = hdul[1].header
        self.energy_center = np.log10(
            (self.data["ENERG_HI"][0] + self.data["ENERG_LO"][0]) / 2.0
        )
        self.migra_center = np.log10(
            (self.data["MIGRA_HI"][0] + self.data["MIGRA_LO"][0]) / 2.0
        )
        self.zenith = (
            np.cos(self.data["THETA_HI"][0]) + np.cos(self.data["THETA_LO"][0])
        ) / 2.0

    def plot_migration(self, ax=None, zenith_index=None, energy_index=None, **kwargs):
        """Plot energy dispersion for given zenith and true energy.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`, optional
            Axis
        zenith_index : int, optional
            index corresponds to item in zenith list
        energy_index : List, optional
            list of items in true energy axes
        **kwargs : dict
            Keyword arguments forwarded to `~matplotlib.pyplot.plot`
        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            Axis
        """
        ax = plt.gca() if ax is None else ax

        if zenith_index is None:
            zenith_index = int(len(self.zenith) / 2)

        if energy_index is None:
            energy_index = [
                0,
                int(len(self.energy_center) / 2),
                len(self.energy_center) - 1,
            ]

        pre_data = self.data["MATRIX"][0][zenith_index].T

        with quantity_support():
            for i in energy_index:
                disp = pre_data[i]
                label = (
                    r"$\cos(\theta)$"
                    + f"={self.zenith[zenith_index]:.2f}\nlog(E)={self.energy_center[i]:.2f}"
                )
                ax.plot(self.migra_center, disp, label=label, **kwargs)

        ax.set_xlabel(r"Migra $\mu$")
        ax.set_ylabel("Probability density")
        ax.legend(loc="upper left")
        return ax

    def plot_bias(self, ax=None, zenith_index=None, add_cbar=True, **kwargs):
        """Plot PDF as a function of true energy and migration for a given zenith.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`, optional
            Axis
        zenith_index : int, optional
            index corresponds to item in zenith list
        add_cbar : bool, default True
            Add a colorbar to the plot.
        kwargs : dict
            Keyword arguments passed to `~matplotlib.pyplot.pcolormesh`.
        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            Axis
        """

        ax = plt.gca() if ax is None else ax

        if zenith_index is None:
            zenith_index = int(len(self.zenith) / 2)

        X, Y = np.meshgrid(self.energy_center, self.migra_center)
        Z = self.data["MATRIX"][0][zenith_index]

        vmin, vmax = np.nanmin(Z), np.nanmax(Z)
        kwargs.setdefault("cmap", "RdPu")
        kwargs.setdefault("edgecolors", "face")
        kwargs.setdefault("shading", "auto")
        kwargs.setdefault("vmin", vmin)
        kwargs.setdefault("vmax", vmax)

        with quantity_support():
            caxes = ax.pcolormesh(X, Y, Z, **kwargs)

        cos_zen = "{:.2f}".format(self.zenith[zenith_index])
        patch = mpatches.Patch(
            edgecolor="black",
            facecolor=(0.28627450980392155, 0.0, 0.41568627450980394),
            label=r"$ \cos(\theta)$=" + cos_zen,
        )
        ax.legend(handles=[patch], loc="lower left")
        ax.axes.set_xlabel(f"log(E_true) [{self.head['TUNIT1']}]")
        ax.axes.set_ylabel(r"Migra $\mu$")

        if add_cbar:
            label = "Probability density [A.U.]"
            ax.figure.colorbar(caxes, ax=ax, label=label)

        return ax

    def peek(self, figsize=(11, 4)):
        """Quick-look summary plots for Edisp.

        Parameters
        ----------
        figsize : tuple
            Size of the figure.
        """
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)
        self.plot_bias(ax=axes[0])
        self.plot_migration(ax=axes[1])
        # self.plot_aeff(ax=axes[2])
        plt.tight_layout()


class DrawPSF:
    """Class is responsible for production of PSF plots."""

    np.seterr(divide="ignore")

    def __init__(self, psf_path=path.join(data_dir, "psf.fits")):
        self.psf_path = psf_path
        with fits.open(self.psf_path) as hdul:
            self.data = hdul[1].data
            self.head = hdul[1].header

        self.energy_center = np.log10(
            (self.data["ENERG_LO"][0] + self.data["ENERG_HI"][0]) / 2.0
        )
        self.energy_logcenter = np.sqrt(
            self.data["ENERG_LO"][0] * self.data["ENERG_HI"][0]
        )
        self.rad_center = (self.data["RAD_HI"][0] + self.data["RAD_LO"][0]) / 2.0

        self.zenith = (
            np.cos(self.data["THETA_HI"][0]) + np.cos(self.data["THETA_LO"][0])
        ) / 2.0

    def interpolate_1d(self):
        values = self.to_psf1D(
            energy=self.energy_logcenter, zenith=np.arccos(self.zenith)
        )
        points = self.rad_center
        return ScaledRegularGridInterpolator(points=points, values=values)

    def interpolate_3d(self):
        energy = self.energy_logcenter
        zenith = np.arccos(self.zenith)
        rad = self.rad_center

        return ScaledRegularGridInterpolator(
            points=(rad, zenith, energy), values=self.data["RPSF"][0]
        )

    def evaluate_1d(self, rad=None):
        r"""Evaluate PSF.

        The following PSF quantities are available:

        * 'dp_domega': PDF per 2-dim solid angle :math:`\Omega` in sr^-1

            .. math:: \frac{dP}{d\Omega}


        Parameters
        ----------
        rad : `~astropy.coordinates.Angle`
            Offset wrt source position

        Returns
        -------
        psf_value : `~astropy.units.Quantity`
            PSF value
        """
        if rad is None:
            rad = self.rad_center

        rad = np.atleast_1d(rad)

        interpolator = self.interpolate_1d()
        return interpolator((rad,))

    def evaluate_3d(self, energy=None, zenith=None, rad=None):
        """Interpolate PSF value at a given zenith and energy.

        Parameters
        ----------
        energy : `~astropy.units.Quantity`
            energy value
        zenith : `~astropy.coordinates.Angle`
            Offset in the field of view
        rad : `~astropy.coordinates.Angle`
            Offset wrt source position

        Returns
        -------
        values : `~astropy.units.Quantity`
            Interpolated value
        """
        if energy is None:
            energy = self.energy_logcenter
        if zenith is None:
            zenith = np.arccos(self.zenith)
        if rad is None:
            rad = self.rad_center

        rad = np.atleast_1d(rad)
        zenith = np.atleast_1d(zenith)
        energy = np.atleast_1d(energy)

        interpolator = self.interpolate_3d()

        return interpolator(
            (
                rad[:, np.newaxis, np.newaxis],
                zenith[np.newaxis, :, np.newaxis],
                energy[np.newaxis, np.newaxis, :],
            )
        )

    def interpolate_containment(self):
        if self.rad_center[0] > 0:
            rad = np.insert(self.rad_center, 0, 0)
        else:
            rad = self.rad_center

        rad_drad = 2 * np.pi * rad * self.evaluate_1d(rad)
        # values = sp.integrate.cumtrapz(
        #     rad_drad.to_value("rad-1"), rad.to_value("rad"), initial=0
        # )
        values = sp.integrate.cumtrapz(rad_drad, rad, initial=0)

        return ScaledRegularGridInterpolator(points=(rad,), values=values, fill_value=1)

    def to_psf1D(self, energy, zenith):
        """Create `~gammapy.irf.TablePSF` at one given energy.

        Parameters
        ----------
        energy : `~astropy.units.Quantity`
            Energy
        theta : `~astropy.coordinates.Angle`
            Offset in the field of view. Default theta = 0 deg

        Returns
        -------
        psf : `~gammapy.irf.TablePSF`
            Table PSF
        """
        # energy = u.Quantity(energy)
        # theta = Angle(theta)
        psf_value = self.evaluate_3d(energy, zenith).squeeze()
        # rad = self.rad_center
        return psf_value

    def plot_psf_vs_rad(self, ax=None, **kwargs):
        """Plot PSF vs rad.

        Parameters
        ----------
        ax : ``

        kwargs : dict
            Keyword arguments passed to `matplotlib.pyplot.plot`
        """

        ax = plt.gca() if ax is None else ax

        bin_edges = np.append(self.data["RAD_LO"][0], self.data["RAD_HI"][0][-1])

        kwargs.setdefault("edgecolor", "black")
        kwargs.setdefault("color", "red")

        with quantity_support():
            ax.hist(
                self.rad_center,
                bins=bin_edges,
                weights=np.sum(self.data["RPSF"][0], axis=(1, 2)),
                **kwargs,
            )
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_xlabel(f"Rad [{self.head['TUNIT5']}]")
        ax.set_ylabel(f"PSF [{self.head['TUNIT7']}]")

        return ax

    def peek(self, figsize=(15, 4)):
        """Quick-look summary plots for PSF.

        Parameters
        ----------
        figsize : tuple
            Size of the figure.
        """
        pass
