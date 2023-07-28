#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A set of functions which are necessary for calculation of IRF

"""
import numpy as np
import pandas as pd
import numba as nb
from numba import jit, njit, prange, float64, int64
from km3pipe.math import azimuth, zenith
import astropy.coordinates as ac
from astropy.time import Time
import astropy.units as u


def calc_theta(table, mc=True):
    """
    Calculate the zenith angle theta of events given the direction coordinates x,y,z.

    Parameters
    ----------
    table : pandas.DataFrame
        Dataframe with the events. Needs to have the keys dir_x, dir_y, dir_z for the local event directions.
    mc : bool, default True
        wether to use the MonteCarlo (true) or the reconstructed directions (false).
        The true directions need to have a "_mc" appended to the directions

    Returns
    -------
    Array
        of zenith angles in rad
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


def edisp_3D(e_bins, m_bins, t_bins, dataset, weights):
    """
    Calculate the 3-dimensional energy dispersion matrix.
    This is just a historgram with the simulated events.
    The normalization needs to be done afterwards.

    Parameters
    ----------
    e_bins : Array
        of energy bins in GeV
    m_bins : Array
        of energy migration bins (reconstructed energy / true energy)
    t_bins : Array
        of zenith angle bins in rad
    dataset : pandas.DataFrame
        with the events
    weights : Array
        of weights for each event

    Returns
    -------
    3D energy dispersion matrix
        binned in energy, energy migration and zenith angle

    """

    if "theta_mc" not in dataset.columns:
        dataset["theta_mc"] = calc_theta(dataset, mc=True)
    if "migra" not in dataset.columns:
        dataset["migra"] = dataset.E / dataset.E_mc

    theta_bins = np.searchsorted(t_bins, dataset.theta_mc) - 1
    energy_bins = np.searchsorted(e_bins, dataset.E_mc) - 1
    migra_bins = np.searchsorted(m_bins, dataset.migra) - 1

    edisp = fill_edisp_3D(
        e_bins, m_bins, t_bins, energy_bins, migra_bins, theta_bins, weights
    )

    return edisp


@njit(fastmath=True, parallel=True)
def fill_edisp_3D(e_bins, m_bins, t_bins, energy_bins, migra_bins, theta_bins, weights):
    """
    numba accelerated helper function to fill the events into the energy disperaion matrix.

    """

    num_t_bins = len(t_bins) - 1
    num_m_bins = len(m_bins) - 1
    num_e_bins = len(e_bins) - 1

    hist = np.zeros((num_t_bins, num_m_bins, num_e_bins), dtype=np.float64)

    num_events = len(energy_bins)
    for i in prange(num_events):
        t_idx = theta_bins[i]
        m_idx = migra_bins[i]
        e_idx = energy_bins[i]
        if (
            0 <= e_idx < num_e_bins
            and 0 <= m_idx < num_m_bins
            and 0 <= t_idx < num_t_bins
        ):
            hist[t_idx, m_idx, e_idx] += weights[i]

    return hist


def psf_3D(e_bins, r_bins, t_bins, dataset, weights):
    """
    Calculate the 3-dimensional PSF matrix.
    This is a historgram with the simulated evenets.
    The normalization needs to be done afterwards.

    Parameters
    ----------
    e_bins : Array
        of energy bins in GeV
    r_bins : Array
        of radial bins in deg (angle between true and reconstructed direction)
    t_bins : Array
        of zenith angle bins in rad
    dataset : pandas.DataFrame
        with the events
    weights : Array
        of weights for each event

    Returns
    -------
    3D PSF matrix
        binned in energy, zenith angle and radial bins

    """

    if "theta_mc" not in dataset:
        dataset["theta_mc"] = calc_theta(dataset, mc=True)

    scalar_prod = (
        dataset.dir_x * dataset.dir_x_mc
        + dataset.dir_y * dataset.dir_y_mc
        + dataset.dir_z * dataset.dir_z_mc
    )
    scalar_prod = np.clip(scalar_prod, -1.0, 1.0)
    rad = np.arccos(scalar_prod) * 180 / np.pi  # in deg now

    theta_bins = np.searchsorted(t_bins, dataset.theta_mc) - 1
    energy_bins = np.searchsorted(e_bins, dataset.E_mc) - 1
    rad_bins = np.searchsorted(r_bins, rad) - 1

    psf = fill_psf_3D(
        e_bins,
        r_bins,
        t_bins,
        energy_bins,
        rad_bins,
        theta_bins,
        weights,
    )

    return psf


@njit(fastmath=True, parallel=True)
def fill_psf_3D(
    e_bins,
    r_bins,
    t_bins,
    energy_bins,
    rad_bins,
    theta_bins,
    weights,
):
    """
    numba accelerated helper function to calculate point spread function.

    """
    num_e_bins = len(e_bins) - 1
    num_r_bins = len(r_bins) - 1
    num_t_bins = len(t_bins) - 1
    hist = np.zeros((num_r_bins, num_t_bins, num_e_bins), dtype=np.float64)

    num_events = len(energy_bins)
    for i in prange(num_events):
        e_idx = energy_bins[i]
        r_idx = rad_bins[i]
        t_idx = theta_bins[i]
        if (
            0 <= e_idx < num_e_bins
            and 0 <= r_idx < num_r_bins
            and 0 <= t_idx < num_t_bins
        ):
            hist[r_idx, t_idx, e_idx] += weights[i]

    return hist


def aeff_2D(e_bins, t_bins, dataset, gamma=1.4, nevents=2e7):
    """
    Calculate the effective area in energy and zenith angle bins.

    Parameters
    ----------
    e_bins : Array
        of energy bins in GeV
    t_bins : Array
        of zenith angle bins in rad
    dataset : pandas.DataFrame
        with the events
    gamma : float, default 1.4
        spectral index of simulated events
    nevents : float, default 2e7
        number of generated events

    Returns
    -------
    2D-Array
        with the effective area in m^2 binned in energy and zenith angle bins

    """

    if "theta_mc" not in dataset:
        dataset["theta_mc"] = calc_theta(dataset, mc=True)

    theta_bins = np.searchsorted(t_bins, dataset.theta_mc) - 1
    energy_bins = np.searchsorted(e_bins, dataset.E_mc) - 1

    w2 = dataset.weight_w2.to_numpy()
    E = dataset.E_mc.to_numpy()
    aeff = fill_aeff_2D(e_bins, t_bins, energy_bins, theta_bins, w2, E, gamma, nevents)

    return aeff


@njit(fastmath=True, parallel=True)
def fill_aeff_2D(e_bins, t_bins, energy_bins, theta_bins, w2, E, gamma, nevents):
    """
    numba accelerated helper function to calculate effective area.

    """
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
