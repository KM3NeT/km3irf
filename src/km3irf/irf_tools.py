#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import numba as nb
from numba import jit, prange
from km3pipe.math import azimuth, zenith
import astropy.coordinates as ac
from astropy.time import Time
import astropy.units as u


def calc_theta(table, mc=True):
    """
    Calculate the zenith angle theta of events given the direction coordinates x,y,z

    table: pandas.Dataframe with the events. Needs to have the keys dir_x, dir_y, dir_z for the local event directions.

    mc: wether to use the MonteCarlo (true) or the reconstructed directions (false).
        The true directions need to have a "_mc" appended to the directions

    Returns: Array of zenith angles in rad

    """

    if not mc:
        dir_x = table["dir_x"].to_numpy()
        dir_y = table["dir_y"].to_numpy()
        dir_z = table["dir_z"].to_numpy()
    else:
        dir_x = table["dir_x_mc"].to_numpy()
        dir_y = table["dir_y_mc"].to_numpy()
        dir_z = table["dir_z_mc"].to_numpy()

    nu_directions = np.vstack([dir_x, dir_y, dir_z]).T

    theta = zenith(nu_directions)  # zenith angles in rad [0:pi]

    return theta


def edisp_3D(e_bins, m_bins, t_bins, dataset, weights=1):
    """
    Calculate the 3-dimensional energy dispersion matrix.
    This is just a historgram with the simulated events.
    The normalization needs to be done afterwards.

    e_bins: Array of energy bins in GeV

    m_bins: Array of energy migration bins (reconstructed energy / true energy)

    t_bins: Array of zenith angle bins in rad

    dataset: pandas.Dataframe with the events

    weights: Array of weights for each event

    Returns: 3-dimensional energy dispersion matrix. Binned in energy, energy migration and zenith angle

    """

    if "theta_mc" not in dataset.keys():
        dataset["theta_mc"] = calc_theta(dataset, mc=True)
    if "migra" not in dataset.keys():
        dataset["migra"] = dataset.E / dataset.E_mc

    theta_bins = pd.cut(dataset.theta_mc, t_bins, labels=False).to_numpy()
    energy_bins = pd.cut(dataset.E_mc, e_bins, labels=False).to_numpy()
    migra_bins = pd.cut(dataset.migra, m_bins, labels=False).to_numpy()

    edisp = fill_edisp_3D(
        e_bins, m_bins, t_bins, energy_bins, migra_bins, theta_bins, weights
    )

    return edisp


@jit(nopython=True, fastmath=False, parallel=True)
def fill_edisp_3D(e_bins, m_bins, t_bins, energy_bins, migra_bins, theta_bins, weights):
    """
    numba accelerated helper function to fill the events into the energy disperaion matrix.
    Needed because numba does not work with pandas but needs numpy arrays.
    fastmath is disabled because it gives different results.

    """

    edisp = np.zeros((len(t_bins) - 1, len(m_bins) - 1, len(e_bins) - 1))
    for i in prange(len(t_bins) - 1):
        for j in range(len(m_bins) - 1):
            for k in range(len(e_bins) - 1):
                mask = (energy_bins == k) & (migra_bins == j) & (theta_bins == i)
                edisp[i, j, k] = np.sum(mask * weights)

    return edisp


def psf_3D(e_bins, r_bins, t_bins, dataset, weights=1):
    """
    Calculate the 3-dimensional PSF matrix.
    This is a historgram with the simulated evenets.
    The normalization needs to be done afterwards.

    e_bins: Array of energy bins in GeV

    r_bins: Array of radial bins in deg (angle between true and reconstructed direction)

    t_bins: Array of zenith angle bins in rad

    dataset: pandas.Dataframe with the events

    weights: Array of weights for each event

    Returns: 3-dimensional PSF matrix. Binned in energy, zenith angle and radial bins

    """

    if "theta_mc" not in dataset.keys():
        dataset["theta_mc"] = calc_theta(dataset, mc=True)

    scalar_prod = (
        dataset.dir_x * dataset.dir_x_mc
        + dataset.dir_y * dataset.dir_y_mc
        + dataset.dir_z * dataset.dir_z_mc
    )
    scalar_prod[scalar_prod > 1.0] = 1.0
    rad = np.arccos(scalar_prod) * 180 / np.pi  # in deg now
    dataset["rad"] = rad

    theta_bins = pd.cut(dataset.theta_mc, t_bins, labels=False).to_numpy()
    energy_bins = pd.cut(dataset.E_mc, e_bins, labels=False).to_numpy()
    rad_bins = pd.cut(rad, r_bins, labels=False).to_numpy()

    psf = fill_psf_3D(
        e_bins, r_bins, t_bins, energy_bins, rad_bins, theta_bins, weights
    )

    return psf


@jit(
    nopython=True, fastmath=False, parallel=True
)  # do not use fastmath=True -- gives 300 aditional entries and no gain
def fill_psf_3D(e_bins, r_bins, t_bins, energy_bins, rad_bins, theta_bins, weights):
    psf = np.zeros((len(r_bins) - 1, len(t_bins) - 1, len(e_bins) - 1))
    for j in prange(len(r_bins) - 1):
        for i in range(len(t_bins) - 1):
            for k in range(len(e_bins) - 1):
                mask = (energy_bins == k) & (rad_bins == j) & (theta_bins == i)
                psf[j, i, k] = np.sum(mask * weights)

    return psf


def aeff_2D(e_bins, t_bins, dataset, gamma=1.4, nevents=2e7):
    """
    Calculate the effective area in energy and zenith angle bins.

    e_bins: Array of energy bins in GeV

    t_bins: Array of zenith angle bins in rad

    dataset: pandas.Dataframe with the events

    gamma: spectral index of simulated events

    nevents: number of generated events

    Returns: 2D-Array with the effective area in m^2 binned in energy and zenith angle bins

    """

    if "theta_mc" not in dataset.keys():
        dataset["theta_mc"] = calc_theta(dataset, mc=True)

    theta_bins = pd.cut(dataset.theta_mc, t_bins, labels=False).to_numpy()
    energy_bins = pd.cut(dataset.E_mc, e_bins, labels=False).to_numpy()

    w2 = dataset.weight_w2.to_numpy()
    E = dataset.E_mc.to_numpy()
    aeff = fill_aeff_2D(e_bins, t_bins, energy_bins, theta_bins, w2, E, gamma, nevents)

    return aeff


@jit(nopython=True, fastmath=False, parallel=True)
def fill_aeff_2D(e_bins, t_bins, energy_bins, theta_bins, w2, E, gamma, nevents):
    T = 365 * 24 * 3600
    aeff = np.empty((len(e_bins) - 1, len(t_bins) - 1))
    for k in prange(len(e_bins) - 1):
        for i in range(len(t_bins) - 1):
            mask = (energy_bins == k) & (theta_bins == i)
            d_omega = -(np.cos(t_bins[i + 1]) - np.cos(t_bins[i]))
            d_E = (e_bins[k + 1]) ** (1 - gamma) - (e_bins[k]) ** (1 - gamma)
            aeff[k, i] = (
                (1 - gamma)
                * np.sum(E[mask] ** (-gamma) * w2[mask])
                / (T * d_omega * d_E * nevents * 2 * np.pi)
            )

    return aeff
