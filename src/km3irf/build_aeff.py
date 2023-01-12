#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

# import awkward as ak
import pandas as pd
import uproot as ur

from km3io import OfflineReader
from .irf_tools import aeff_2D

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from astropy.io import fits
import astropy.units as u
from astropy.io import fits

from scipy.stats import binned_statistic


# from collections import defaultdict

# import sys
# sys.path.append('../')
# from python_scripts.irf_utils import aeff_2D, psf_3D
# from python_scripts.func import get_cut_mask
# from python_scripts.func import WriteAeff
# from python_scripts.func import WritePSF


class DataContainer:
    def __init__(self, infile, no_bdt=False):
        self.f_km3io = OfflineReader(infile)
        self.f_uproot = ur.open(infile)
        self.df = unpack_data(no_bdt, self.f_uproot)

    def apply_cuts(self):
        """
        Apply cuts to the created data frame

        """
        mask = get_cut_mask(self.df.bdt0, self.df.bdt1, self.df.dir_z)
        self.df = self.df[mask].copy()
        return None
        # df_cut = self.df[mask].copy()
        # return df_cut

    def weight_calc(self, tag, df_pass, weight_factor=-2.5):
        """
        calculate the normalized weight factor for each event

        tag: "nu" or "nubar"

        df_pass: incoming data frame

        weight_factor: re-weight data, default value  -2.5

        """
        alpha_value = self.f_km3io.header.spectrum.alpha
        weights = dict()
        weights[tag] = (df_pass.E_mc ** (weight_factor - alpha_value)).to_numpy()
        weights[tag] *= len(df_pass) / weights[tag].sum()
        return weights

    def merge_flavors(self, df_nu, df_nubar):
        """
        Merge two data frames with differnt flavors in one

        df_nu: data frame for 'nu'

        df_nubar: data frame for 'nubar'

        return the merged pandas data frame

        """
        df_merged = pd.concat([df_nu, df_nubar], ignore_index=True)
        return df_merged

    def build_aeff(
        self,
        df_pass,
        weight_factor=-2.5,
        cos_theta_binE=np.linspace(1, -1, 13),
        energy_binE=np.logspace(2, 8, 49),
        output="aeff.fits",
    ):
        """
        Build Effective Area 2D .fits

        df_pass: incoming data frame

        weight_factor: re-weight data, default value  -2.5

        cos_theta_binE: numpy array of linear bins for cos of zenith angle theta

        energy_binE: log numpy array of enegy bins

        output: name of generated Aeff file with extension .fits

        """
        theta_binE = np.arccos(cos_theta_binE)
        # Bin centers
        energy_binC = np.sqrt(energy_binE[:-1] * energy_binE[1:])
        theta_binC = np.arccos(0.5 * (cos_theta_binE[:-1] + cos_theta_binE[1:]))

        # Fill histograms for effective area
        # !!! Check number of events uproot vs pandas
        aeff_all = (
            aeff_2D(
                e_bins=energy_binE,
                t_bins=theta_binE,
                dataset=df_pass,
                gamma=(-weight_factor),
                nevents=df_pass.shape[0],
            )
            * 2
        )  # two building blocks

        new_aeff_file = WriteAeff(
            energy_binC, energy_binE, theta_binC, theta_binE, aeff_hist=aeff_all
        )
        new_aeff_file.to_fits(file_name=output)

        return None


def unpack_data(no_bdt, uproot_file):
    """
    retrieve information from data and pack it to pandas DataFrame

    uproot_file: input uproot file

    return pandas data frame

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
    bdt0: to determine groups to which BDT cut should be applied (upgoing/horizontal/downgoing)

    bdt1: BDT score in the range [-1, 1]. Closer to 1 means more signal-like

    dir_z: is the reconstructed z-direction of the event

    return a mask for set cuts

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
    def __init__(self, energy_binC, energy_binE, theta_binC, theta_binE, aeff_hist):
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
            array=[aeff_hist],
        )

    def to_fits(self, file_name):
        """
        write Aeff to .fits file

        file_name: should have .fits extension

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
