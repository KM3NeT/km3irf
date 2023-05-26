#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
It allows to produce different .fits files from original data dst.root files , which are required  for IRF.

"""
import numpy as np

# import awkward as ak
import pandas as pd
import uproot as ur

from km3io import OfflineReader
from .irf_tools import aeff_2D, psf_3D, edisp_3D

import matplotlib.pyplot as plt
from matplotlib import cm

# from matplotlib.colors import LogNorm

from astropy.io import fits
import astropy.units as u
from astropy.visualization import quantity_support

from scipy.stats import binned_statistic
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from .utils import data_dir
from os import path

# from collections import defaultdict

# import sys
# sys.path.append('../')
# from python_scripts.irf_utils import aeff_2D, psf_3D
# from python_scripts.func import get_cut_mask
# from python_scripts.func import WriteAeff
# from python_scripts.func import WritePSF


class DataContainer:
    """
    General class, which allows operating with IRF components such as:
    Effective area (aeff), Point spread function (psf), Energy dispersion (edisp).

    Attributes
    ----------
    infile : root
        input dst.root file with data
    no_bdt : bool
        with/out bdt information in input file

    """

    def __init__(self, infile, no_bdt=False):
        self.f_km3io = OfflineReader(infile)
        self.f_uproot = ur.open(infile)
        self.df = unpack_data(no_bdt, self.f_uproot)

    def apply_cuts(self):
        """
        Apply cuts to the created data frame.

        Returns
        -------
        None

        """
        mask = get_cut_mask(self.df.bdt0, self.df.bdt1, self.df.dir_z)
        self.df = self.df[mask].copy()
        return None
        # df_cut = self.df[mask].copy()
        # return df_cut

    def weight_calc(self, tag, weight_factor=-2.5):
        r"""
        Calculate the normalized weight factor for each event.

        Parameters
        ----------
        tag : str
            poddible options "nu" or "nubar"
        weight_factor : float, default -2.5
            spectral index for re-weight data

        Returns
        -------
        weights : Array

        """
        try:
            alpha_value = self.f_km3io.header.spectrum.alpha

        except AttributeError:
            print("your input data file has no header, alpha_value set to default -1.4")
            alpha_value = -1.4

        weights = dict()
        weights[tag] = (self.df.E_mc ** (weight_factor - alpha_value)).to_numpy()
        weights[tag] *= len(self.df) / weights[tag].sum()
        return weights

    # @staticmethod
    def merge_flavors(self, df_flavor):
        """
        Merge two data frames with differnt flavors in one.

        Parameters
        ----------
        df_flavor : pandas.DataFrame
            data frame with another flavor 'nu' or 'anu'

        Returns
        -------
        None
        """
        self.df = pd.concat([self.df, df_flavor], ignore_index=True)

    def build_aeff(
        self,
        weight_factor=-2.5,
        cos_theta_binE=np.linspace(1, -1, 13),
        energy_binE=np.logspace(2, 8, 49),
        output="aeff.fits",
    ):
        """
        Build Effective Area 2D .fits file.

        Parameters
        ----------
        weight_factor : float, default -2.5
            re-weight input data with new spectral index
        cos_theta_binE : Array, default np.linspace(1, -1, 13)
            of linear bins for cos of zenith angle theta
        energy_binE : Array, default np.logspace(2, 8, 49)
            log numpy array of enegy bins
        output : str, default "aeff.fits"
            name of generated Aeff file with extension .fits

        Returns
        -------
        None

        """
        theta_binE = np.arccos(cos_theta_binE)
        # Bin centers
        energy_binC = np.sqrt(energy_binE[:-1] * energy_binE[1:])
        theta_binC = np.arccos(0.5 * (cos_theta_binE[:-1] + cos_theta_binE[1:]))

        # Fill histograms for effective area
        aeff_all = (
            aeff_2D(
                e_bins=energy_binE,
                t_bins=theta_binE,
                dataset=self.df,
                gamma=(-weight_factor),
                nevents=self.df.shape[0],
            )
            * 2
        )  # two building blocks

        new_aeff_file = WriteAeff(
            energy_binC, energy_binE, theta_binC, theta_binE, aeff_T=aeff_all
        )
        new_aeff_file.to_fits(file_name=output)

        return None

    def build_psf(
        self,
        cos_theta_binE=np.linspace(1, -1, 7),
        energy_binE=np.logspace(2, 8, 25),
        rad_binE=np.concatenate(
            (
                np.linspace(0, 1, 20, endpoint=False),
                np.linspace(1, 5, 40, endpoint=False),
                np.linspace(5, 30, 51),
                [180.0],
            )
        ),
        norm=False,
        smooth=True,
        smooth_norm=True,
        output="psf.fits",
    ):
        """
        Build Point Spread Function 3D .fits file.

        Parameters
        ----------
        cos_theta_binE : Array, default np.linspace(1, -1, 7)
            of linear bins for cos of zenith angle theta
        energy_binE : Array, default np.logspace(2, 8, 25)
            log numpy array of enegy bins
        rad_binE : Array
            of linear radial bins (20 bins for 0-1 deg, 40 bins for 1-5 deg,
            50 bins for 5-30 deg, + 1 final bin up to 180 deg)
        norm : bool, default False
            enable or disable normalization
        smooth : bool, default True
            enable or disable smearing
        smooth_norm : bool, default True
            enable or disable smearing with normalization,
            can't be the same with norm
        output : str, default "psf.fits"
            name of generated PSF file with extension .fits

        Returns
        -------
        None

        """
        theta_binE = np.arccos(cos_theta_binE)
        # Bin centers
        energy_binC = np.sqrt(energy_binE[:-1] * energy_binE[1:])
        theta_binC = np.arccos(0.5 * (cos_theta_binE[:-1] + cos_theta_binE[1:]))
        rad_binC = 0.5 * (rad_binE[1:] + rad_binE[:-1])

        # Fill histogram for PSF
        psf = psf_3D(
            e_bins=energy_binE,
            r_bins=rad_binE,
            t_bins=theta_binE,
            dataset=self.df,
            weights=1,
        )

        # compute dP/dOmega
        sizes_rad_binE = np.diff(rad_binE**2)
        psf /= sizes_rad_binE[:, None, None] * (np.pi / 180) ** 2 * np.pi

        # Normalization for PSF
        if norm:
            psf = np.nan_to_num(psf / psf.sum(axis=0, keepdims=True))

        # Smearing
        if smooth and not norm:
            s1 = gaussian_filter1d(psf, 0.5, axis=0, mode="nearest")
            s2 = gaussian_filter1d(psf, 2, axis=0, mode="nearest")
            s3 = gaussian_filter1d(psf, 4, axis=0, mode="nearest")
            s4 = gaussian_filter1d(psf, 6, axis=0, mode="constant")
            psf = np.concatenate(
                (s1[:10], s2[10:20], s3[20:60], s4[60:-1], [psf[-1]]), axis=0
            )
            # smooth edges between the different ranges
            psf[10:-1] = gaussian_filter1d(psf[10:-1], 1, axis=0, mode="nearest")
            if smooth_norm:
                norm_psf_sm = (
                    psf * sizes_rad_binE[:, None, None] * (np.pi / 180) ** 2 * np.pi
                ).sum(axis=0, keepdims=True)
                psf = np.nan_to_num(psf / norm_psf_sm)
        elif smooth and norm:
            raise Exception("smooth and norm cannot be True at the same time")

        new_psf_file = WritePSF(
            energy_binC,
            energy_binE,
            theta_binC,
            theta_binE,
            rad_binC,
            rad_binE,
            psf=psf,
        )
        new_psf_file.to_fits(file_name=output)

        return None

    def build_edisp(
        self,
        cos_theta_binE=np.linspace(1, -1, 7),
        energy_binE=np.logspace(2, 8, 25),
        migra_binE=np.logspace(-5, 2, 57),
        norm=False,
        smooth=True,
        smooth_norm=True,
        output="edisp.fits",
    ):
        """
        Build Energy dispertion 3D .fits file.

        Parameters
        ----------
        cos_theta_binE : Array, default np.linspace(1, -1, 7)
            of linear bins for cos of zenith angle theta
        energy_binE : Array, default np.logspace(2, 8, 25)
            log numpy array of enegy bins
        migra_binE : Array, default np.logspace(-5, 2, 57)
            log numpy array of migration enegy bins
        norm : bool, default False
            enable or disable normalization
        smooth : bool, default True
            enable or disable smearing
        smooth_norm : bool, default True
            enable or disable smearing with normalization,
            can't be the same with norm
        output : str, default "edisp.fits"
            name of generated Edisp file with extension .fits

        Returns
        -------
        None

        """
        theta_binE = np.arccos(cos_theta_binE)
        # Bin centers
        energy_binC = np.sqrt(energy_binE[:-1] * energy_binE[1:])
        theta_binC = np.arccos(0.5 * (cos_theta_binE[:-1] + cos_theta_binE[1:]))
        migra_binC = np.sqrt(migra_binE[:-1] * migra_binE[1:])

        # fill histogram for Edisp
        edisp = edisp_3D(
            e_bins=energy_binE,
            m_bins=migra_binE,
            t_bins=theta_binE,
            dataset=self.df,
            weights=1,
        )

        sizes_migra_binE = np.diff(migra_binE)
        edisp /= sizes_migra_binE[:, np.newaxis]
        m_normed = edisp * sizes_migra_binE[:, np.newaxis]

        if norm:
            edisp = np.nan_to_num(edisp / m_normed.sum(axis=1, keepdims=True))

        # Smearing
        if smooth and not norm:
            for i in range(edisp.shape[-1]):
                for j in range(edisp.shape[0]):
                    kernel_size = 2 - 0.25 * max(0, np.log10(edisp[j, :, i].sum()))
                    edisp[j, :, i] = gaussian_filter1d(
                        edisp[j, :, i] * sizes_migra_binE,
                        kernel_size,
                        axis=0,
                        mode="nearest",
                    )
            edisp /= sizes_migra_binE[:, None]
            if smooth_norm:
                m_normed = edisp * sizes_migra_binE[:, np.newaxis]
                edisp = np.nan_to_num(edisp / m_normed.sum(axis=1, keepdims=True))
        elif smooth and norm:
            raise Exception("smooth and norm cannot be True at the same time")

        new_edisp_file = WriteEdisp(
            energy_binC,
            energy_binE,
            theta_binC,
            theta_binE,
            migra_binC,
            migra_binE,
            edisp=edisp,
        )
        new_edisp_file.to_fits(file_name=output)

        return None


def unpack_data(no_bdt, uproot_file):
    """
    Retrieve information from data file and pack it to pandas.DataFrame.

    Parameters
    ----------
    no_bdt : bool
       with/out bdt information in input file
    uproot_file : uproot.open()
        input uproot file

    Returns
    -------
    pandas.DataFrame

    """
    # Access data arrays
    data_uproot = dict()

    E_evt = uproot_file["E/Evt"]

    data_uproot["E"] = E_evt["trks/trks.E"].array()[:, 0]
    data_uproot["dir_x"] = E_evt["trks/trks.dir.x"].array()[:, 0]
    data_uproot["dir_y"] = E_evt["trks/trks.dir.y"].array()[:, 0]
    data_uproot["dir_z"] = E_evt["trks/trks.dir.z"].array()[:, 0]

    data_uproot["E_mc"] = E_evt["mc_trks/mc_trks.E"].array()[:, 0]
    data_uproot["dir_x_mc"] = E_evt["mc_trks/mc_trks.dir.x"].array()[:, 0]
    data_uproot["dir_y_mc"] = E_evt["mc_trks/mc_trks.dir.y"].array()[:, 0]
    data_uproot["dir_z_mc"] = E_evt["mc_trks/mc_trks.dir.z"].array()[:, 0]
    data_uproot["weight_w2"] = E_evt["w"].array()[:, 1]

    # extracting bdt information
    if not no_bdt:
        T = uproot_file["T"]
        bdt = T["bdt"].array()
        data_uproot["bdt0"] = bdt[:, 0]
        data_uproot["bdt1"] = bdt[:, 1]

    # create Data Frames
    df_data = pd.DataFrame(data_uproot)

    return df_data


def get_cut_mask(bdt0, bdt1, dir_z):
    """
    Create a cut mask for chosen cuts

    Parameters
    ----------
    bdt0 : int
        to determine groups to which BDT cut should be applied
        (upgoing/horizontal/downgoing)
    bdt1 : float
        BDT score in the range [-1, 1]. Closer to 1 means more signal-like
    dir_z : float
        the reconstructed z-direction of the event

    Returns
    -------
    Array(bool)

    """

    mask_down = bdt0 >= 11  # remove downgoing events
    clear_signal = bdt0 == 12  # very clear signal
    loose_up = (np.arccos(dir_z) * 180 / np.pi < 80) & (
        bdt1 > 0.0
    )  # apply loose cut on upgoing events
    strong_horizontal = (np.arccos(dir_z) * 180 / np.pi > 80) & (
        bdt1 > 0.7
    )  # apply strong cut on horizontal events

    return mask_down & (clear_signal | loose_up | strong_horizontal)


# Class for writing aeff_2D to fits files
class WriteAeff:
    """Class with defenitions of headers for Aeff .fits file."""

    def __init__(self, energy_binC, energy_binE, theta_binC, theta_binE, aeff_T):
        self.col1 = fits.Column(
            name="ENERG_LO",
            format="{}E".format(len(energy_binC)),
            unit="GeV",
            array=[energy_binE[:-1]],
        )
        self.col2 = fits.Column(
            name="ENERG_HI",
            format="{}E".format(len(energy_binC)),
            unit="GeV",
            array=[energy_binE[1:]],
        )
        self.col3 = fits.Column(
            name="THETA_LO",
            format="{}E".format(len(theta_binC)),
            unit="rad",
            array=[theta_binE[:-1]],
        )
        self.col4 = fits.Column(
            name="THETA_HI",
            format="{}E".format(len(theta_binC)),
            unit="rad",
            array=[theta_binE[1:]],
        )
        self.col5 = fits.Column(
            name="EFFAREA",
            format="{}D".format(len(energy_binC) * len(theta_binC)),
            dim="({},{})".format(len(energy_binC), len(theta_binC)),
            unit="m2",
            array=[aeff_T.T],
        )

    def to_fits(self, file_name):
        """
        Write Aeff to .fits file.

        Parameters
        ----------
        file_name : str
            should have .fits extension

        """
        cols = fits.ColDefs([self.col1, self.col2, self.col3, self.col4, self.col5])
        hdu = fits.PrimaryHDU()
        hdu2 = fits.BinTableHDU.from_columns(cols)
        hdu2.header["EXTNAME"] = "EFFECTIVE AREA"
        hdu2.header[
            "HDUDOC"
        ] = "https://github.com/open-gamma-ray-astro/gamma-astro-data-formats"
        hdu2.header["HDUVERS"] = "0.2"
        hdu2.header["HDUCLASS"] = "GADF"
        hdu2.header["HDUCLAS1"] = "RESPONSE"
        hdu2.header["HDUCLAS2"] = "EFF_AREA"
        hdu2.header["HDUCLAS3"] = "FULL-ENCLOSURE"
        hdu2.header["HDUCLAS4"] = "AEFF_2D"
        aeff_fits = fits.HDUList([hdu, hdu2])
        aeff_fits.writeto(file_name, overwrite=True)

        return print(f"file {file_name} is written successfully!")


# Class for writing PSF to fits files
class WritePSF:
    """Class with defenitions of headers for PSF .fits file."""

    def __init__(
        self,
        energy_binC,
        energy_binE,
        theta_binC,
        theta_binE,
        rad_binC,
        rad_binE,
        psf,
    ):
        self.col1 = fits.Column(
            name="ENERG_LO",
            format="{}E".format(len(energy_binC)),
            unit="GeV",
            array=[energy_binE[:-1]],
        )
        self.col2 = fits.Column(
            name="ENERG_HI",
            format="{}E".format(len(energy_binC)),
            unit="GeV",
            array=[energy_binE[1:]],
        )
        self.col3 = fits.Column(
            name="THETA_LO",
            format="{}E".format(len(theta_binC)),
            unit="rad",
            array=[theta_binE[:-1]],
        )
        self.col4 = fits.Column(
            name="THETA_HI",
            format="{}E".format(len(theta_binC)),
            unit="rad",
            array=[theta_binE[1:]],
        )
        self.col5 = fits.Column(
            name="RAD_LO",
            format="{}E".format(len(rad_binC)),
            unit="deg",
            array=[rad_binE[:-1]],
        )
        self.col6 = fits.Column(
            name="RAD_HI",
            format="{}E".format(len(rad_binC)),
            unit="deg",
            array=[rad_binE[1:]],
        )
        self.col7 = fits.Column(
            name="RPSF",
            format="{}D".format(len(energy_binC) * len(theta_binC) * len(rad_binC)),
            dim="({},{},{})".format(len(energy_binC), len(theta_binC), len(rad_binC)),
            unit="sr-1",
            array=[psf],
        )

    def to_fits(self, file_name):
        """
        Write PSF to .fits file.

        Parameters
        ----------
        file_name : str
            should have .fits extension
        """
        cols = fits.ColDefs(
            [
                self.col1,
                self.col2,
                self.col3,
                self.col4,
                self.col5,
                self.col6,
                self.col7,
            ]
        )
        hdu = fits.PrimaryHDU()
        hdu2 = fits.BinTableHDU.from_columns(cols)
        hdu2.header["EXTNAME"] = "PSF_2D_TABLE"
        hdu2.header[
            "HDUDOC"
        ] = "https://github.com/open-gamma-ray-astro/gamma-astro-data-formats"
        hdu2.header["HDUVERS"] = "0.2"
        hdu2.header["HDUCLASS"] = "GADF"
        hdu2.header["HDUCLAS1"] = "RESPONSE"
        hdu2.header["HDUCLAS2"] = "RPSF"
        hdu2.header["HDUCLAS3"] = "FULL-ENCLOSURE"
        hdu2.header["HDUCLAS4"] = "PSF_TABLE"
        psf_fits = fits.HDUList([hdu, hdu2])
        psf_fits.writeto(file_name, overwrite=True)

        return print(f"file {file_name} is written successfully!")


# Class for writing Edisp to fits files
class WriteEdisp:
    """Class with defenitions of headers for Edisp .fits file."""

    def __init__(
        self,
        e_binc_coarse,
        e_bins_coarse,
        t_binc_coarse,
        t_bins_coarse,
        migra_binc,
        migra_bins,
        edisp,
    ):
        self.col1 = fits.Column(
            name="ENERG_LO",
            format="{}E".format(len(e_binc_coarse)),
            unit="GeV",
            array=[e_bins_coarse[:-1]],
        )
        self.col2 = fits.Column(
            name="ENERG_HI",
            format="{}E".format(len(e_binc_coarse)),
            unit="GeV",
            array=[e_bins_coarse[1:]],
        )
        self.col3 = fits.Column(
            name="MIGRA_LO",
            format="{}E".format(len(migra_binc)),
            array=[migra_bins[:-1]],
        )
        self.col4 = fits.Column(
            name="MIGRA_HI",
            format="{}E".format(len(migra_binc)),
            array=[migra_bins[1:]],
        )
        self.col5 = fits.Column(
            name="THETA_LO",
            format="{}E".format(len(t_binc_coarse)),
            unit="rad",
            array=[t_bins_coarse[:-1]],
        )
        self.col6 = fits.Column(
            name="THETA_HI",
            format="{}E".format(len(t_binc_coarse)),
            unit="rad",
            array=[t_bins_coarse[1:]],
        )
        self.col7 = fits.Column(
            name="MATRIX",
            format="{}D".format(
                len(e_binc_coarse) * len(migra_binc) * len(t_binc_coarse)
            ),
            dim="({},{},{})".format(
                len(e_binc_coarse), len(migra_binc), len(t_binc_coarse)
            ),
            array=[edisp * np.diff(migra_bins)[:, None]],
        )

    def to_fits(self, file_name):
        """
        Write Edisp to .fits file.

        Parameters
        ----------
        file_name : str
            should have .fits extension
        """
        cols = fits.ColDefs(
            [
                self.col1,
                self.col2,
                self.col3,
                self.col4,
                self.col5,
                self.col6,
                self.col7,
            ]
        )
        hdu = fits.PrimaryHDU()
        hdu2 = fits.BinTableHDU.from_columns(cols)
        hdu2.header["EXTNAME"] = "EDISP_2D"
        hdu2.header[
            "HDUDOC"
        ] = "https://github.com/open-gamma-ray-astro/gamma-astro-data-formats"
        hdu2.header["HDUVERS"] = "0.2"
        hdu2.header["HDUCLASS"] = "GADF"
        hdu2.header["HDUCLAS1"] = "RESPONSE"
        hdu2.header["HDUCLAS2"] = "EDISP"
        hdu2.header["HDUCLAS3"] = "FULL-ENCLOSURE"
        hdu2.header["HDUCLAS4"] = "EDISP_2D"
        edisp_fits = fits.HDUList([hdu, hdu2])
        edisp_fits.writeto(file_name, overwrite=True)

        return print(f"file {file_name} is written successfully!")
