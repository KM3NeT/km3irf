#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import awkward as ak
import pandas as pd
import uproot as ur
from km3io import OfflineReader

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from astropy.io import fits
import astropy.units as u

# from gammapy.irf import EnergyDispersion2D

from scipy.stats import binned_statistic

# from scipy.ndimage import gaussian_filter1d, gaussian_filter


# from collections import defaultdict

# import sys
# sys.path.append('../')
# from python_scripts.irf_utils import aeff_2D, psf_3D
# from python_scripts.func import get_cut_mask
# from python_scripts.func import WriteAeff
# from python_scripts.func import WritePSF


def build_aeff(
    input=(file_nu, file_nubar), no_bdt=False, cuts=False, power_degree=-2.5
):
    """
    Create Aeff .fits from dist files

    input: a tuple with pathes to nu_dst file and nubar_dst file

    no_bdt: include or exclude bdt, default False

    cuts: apply cuts, default False

    power_degree: re-weight data, default value  -2.5

    """

    # Read data files using km3io
    f_nu_km3io = OfflineReader(input[0])
    f_nubar_km3io = OfflineReader(input[1])

    # Read data files using uproot
    f_nu_uproot = ur.open(input[0])
    f_nubar_uproot = ur.open(input[1])


def unpack_data(no_bdt):
    """
    some words

    return tuple with two pandas data frames (nu, nubar)
    """
    # Access data arrays
    data_km3io = dict()

    for l, f in zip(["nu", "nubar"], [f_nu_km3io, f_nubar_km3io]):
        data_km3io[l] = dict()

        data_km3io[l]["E"] = f.tracks.E[:, 0]
        data_km3io[l]["dir_x"] = f.tracks.dir_x[:, 0]
        data_km3io[l]["dir_y"] = f.tracks.dir_y[:, 0]
        data_km3io[l]["dir_z"] = f.tracks.dir_z[:, 0]

        data_km3io[l]["energy_mc"] = f.mc_tracks.E[:, 0]
        data_km3io[l]["dir_x_mc"] = f.mc_tracks.dir_x[:, 0]
        data_km3io[l]["dir_y_mc"] = f.mc_tracks.dir_y[:, 0]
        data_km3io[l]["dir_z_mc"] = f.mc_tracks.dir_z[:, 0]

        data_km3io[l]["weight_w2"] = f.w[:, 1]

    # extracting bdt information
    if not no_bdt:
        for l, f in zip(["nu", "nubar"], [f_nu_uproot, f_nubar_uproot]):
            T = f["T;1"]
            bdt = T["bdt"].array()
            data_km3io[l]["bdt0"] = bdt[:, 0]
            data_km3io[l]["bdt1"] = bdt[:, 1]

    # create Data Frames
    df_nu = pd.DataFrame(data_km3io["nu"])
    df_nubar = pd.DataFrame(data_km3io["nubar"])

    data_tuple = (df_nu, df_nubar)

    return data_tuple
