#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest
import numpy as np
import astropy.units as u
from astropy.coordinates import Angle
from astropy.table import Table
from astropy.io import fits

from km3irf import psf_3d
from km3irf.test_utils import mpl_plot_check


class TestPSF3D(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)

        self.energy_lo = [0.1, 0.3] * u.TeV
        self.energy_hi = [0.3, 1.0] * u.TeV
        self.theta_lo = [0 * u.rad, 0.5 * u.rad]
        self.theta_hi = [0.5 * u.rad, 1.2 * u.rad]
        self.rad_lo = [0 * u.deg, 2.0 * u.deg]
        self.rad_hi = [2.0 * u.deg, 5.0 * u.deg]
        self.psf_value = np.random.random((2, 2, 2)) * u.Unit("sr^-1")
        self.psf = psf_3d.PSF3D(
            self.energy_lo,
            self.energy_hi,
            self.theta_lo,
            self.theta_hi,
            self.rad_lo,
            self.rad_hi,
            self.psf_value,
        )
        self.psf_original = psf_3d.PSF3D.read()

    def test_psf_3d_init(self):
        assert np.allclose(self.psf.energy_lo, self.energy_lo.to("GeV"))
        assert np.allclose(self.psf.energy_hi, self.energy_hi.to("GeV"))
        assert np.allclose(self.psf.theta_lo, self.theta_lo)
        assert np.allclose(self.psf.theta_hi, self.theta_hi)
        assert np.allclose(self.psf.rad_lo, self.rad_lo)
        assert np.allclose(self.psf.rad_hi, self.rad_hi)
        assert np.allclose(self.psf.psf_value, self.psf_value.to("sr^-1"))

    def test_psf_3d_interpolation(self):
        energy = 0.2 * u.TeV
        theta = 0.4 * u.rad
        rad = 3.0 * u.deg

        interpolated_value = self.psf.evaluate(energy, theta, rad)

        assert np.allclose(interpolated_value.value[0][0][0], 0.4665962)

    def test_psf_3d_containment_radius(self):
        energy = [0.2, 0.5] * u.TeV
        theta = [20, 30] * u.deg

        containment_radius = self.psf.containment_radius(energy, theta)

        assert np.allclose(containment_radius.value[0], [3.5, 3.5])

    def test_plots(self):
        with mpl_plot_check():
            self.psf_original.plot_containment()

        with mpl_plot_check():
            self.psf_original.plot_containment_vs_energy()

        with mpl_plot_check():
            self.psf_original.plot_psf_vs_rad()

        with mpl_plot_check():
            self.psf_original.peek()
