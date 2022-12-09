#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import awkward as ak
import pandas as pd
import uproot as ur
from km3io import OfflineReader
from irf_tools import aeff_2D

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from astropy.io import fits
import astropy.units as u
from astropy.io import fits

# from gammapy.irf import EnergyDispersion2D

from scipy.stats import binned_statistic


# from collections import defaultdict

# import sys
# sys.path.append('../')
# from python_scripts.irf_utils import aeff_2D, psf_3D
# from python_scripts.func import get_cut_mask
# from python_scripts.func import WriteAeff
# from python_scripts.func import WritePSF


def build_aeff(
    input,
    tag="nu",
    no_bdt=False,
    cuts=True,
    weight_factor=-2.5,
    cos_theta_binE=np.linspace(1, -1, 13),
    energy_binE=np.logspace(2, 8, 49),
    output="aeff.fits",
):
    """
    Create Aeff .fits from dist files

    input: a path to nu_dst file or nubar_dst file

    tag: specify neutrino type nu or nubar, default nu

    no_bdt: include or exclude bdt, default False

    cuts: apply cuts, default True

    weight_factor: re-weight data, default value  -2.5

    cos_theta_binE: numpy array of linear bins for cos of zenith angle theta

    energy_binE: log numpy array of enegy bins

    output: name of generated Aeff file with extension .fits

    """

    # Read data files using km3io
    f_km3io = OfflineReader(input)
    # f_nubar_km3io = OfflineReader(input[1])

    # Read data files using uproot
    f_uproot = ur.open(input)
    # f_nubar_uproot = ur.open(input[1])

    df = unpack_data(no_bdt, tag, f_km3io, f_uproot)
    # df_nubar = unpack_data(no_bdt, "nubar", f_nubar_km3io, f_nubar_uproot)

    alpha_value = f_km3io.header.spectrum.alpha

    if cuts:
        mask = get_cut_mask(df.bdt0, df.bdt1, df.dir_z)
        df = df[mask].copy()
        # mask_nubar = get_cut_mask(df_nubar.bdt0, df_nubar.bdt1, df_nubar.dir_z)
        # df_nubar = df_nubar[mask_nubar].copy()

    # calculate the normalized weight factor for each event
    # weights = dict()
    # for l, df in zip(["nu", "nubar"], [df_nu, df_nubar]):
    #     weights[l] = (df.energy_mc ** (weight_factor - alpha_value)).to_numpy()
    #     weights[l] *= len(df) / weights[l].sum()

    # Needs to continue here
    weights = {tag: df}
    for l, df in zip(["nu", "nubar"], [df_nu, df_nubar]):
        weights[l] = (df.energy_mc ** (weight_factor - alpha_value)).to_numpy()
        weights[l] *= len(df) / weights[l].sum()

    # Create DataFrames with neutrinos and anti-neutrinos
    df_nu_all = pd.concat([df_nu, df_nubar], ignore_index=True)

    # Also create a concatenated array for the weights
    weights_all = np.concatenate([weights["nu"], weights["nubar"]])

    # Define bins for Aeff
    # cos_bins_fine = np.linspace(1, -1, 13)
    theta_binE = np.arccos(cos_theta_binE)
    # energy_binE = np.logspace(2, 8, 49)

    # Bin centers
    energy_binC = np.sqrt(energy_binE[:-1] * energy_binE[1:])
    theta_binC = np.arccos(0.5 * (cos_theta_binE[:-1] + cos_theta_binE[1:]))

    # Fill histograms for effective area
    aeff_all = (
        aeff_2D(
            e_bins=energy_binE,
            t_bins=theta_binE,
            dataset=df_nu_all,
            gamma=(-weight_factor),
            nevents=f_nu_km3io.header.genvol.numberOfEvents
            + f_nubar_km3io.header.genvol.numberOfEvents,
        )
        * 2
    )  # two building blocks

    new_aeff_file = WriteAeff(
        energy_binC, energy_binE, theta_binC, theta_binE, aeff_hist=aeff_all
    )
    new_aeff_file.to_fits(file_name=output)


def unpack_data(no_bdt, type, km3io_file, uproot_file):
    """
    retrieve information from data and pack it to DataFrame

    type: "nu" or "nubar"

    km3io_file: input km3io file

    uproot_file: input uproot file

    return pandas data frame for specific type
    """
    # Access data arrays
    data_km3io = dict()

    # for l, f in zip(["nu", "nubar"], [f_nu_km3io, f_nubar_km3io]):
    data_km3io[type] = dict()

    data_km3io[type]["E"] = km3io_file.tracks.E[:, 0]
    data_km3io[type]["dir_x"] = km3io_file.tracks.dir_x[:, 0]
    data_km3io[type]["dir_y"] = km3io_file.tracks.dir_y[:, 0]
    data_km3io[type]["dir_z"] = km3io_file.tracks.dir_z[:, 0]

    data_km3io[type]["energy_mc"] = km3io_file.mc_tracks.E[:, 0]
    data_km3io[type]["dir_x_mc"] = km3io_file.mc_tracks.dir_x[:, 0]
    data_km3io[type]["dir_y_mc"] = km3io_file.mc_tracks.dir_y[:, 0]
    data_km3io[type]["dir_z_mc"] = km3io_file.mc_tracks.dir_z[:, 0]

    data_km3io[type]["weight_w2"] = km3io_file.w[:, 1]

    # extracting bdt information
    if not no_bdt:
        # for l, f in zip(["nu", "nubar"], [f_nu_uproot, f_nubar_uproot]):
        T = uproot_file["T;1"]
        bdt = T["bdt"].array()
        data_km3io[type]["bdt0"] = bdt[:, 0]
        data_km3io[type]["bdt1"] = bdt[:, 1]

    # create Data Frames
    df_data = pd.DataFrame(data_km3io[type])
    # df_nubar = pd.DataFrame(data_km3io["nubar"])

    # data_tuple = (df_nu, df_nubar)

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
        write .fits file

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
