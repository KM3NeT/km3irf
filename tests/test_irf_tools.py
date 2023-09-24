#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest
import numpy as np
import pandas as pd

from km3irf import irf_tools
from km3pipe.math import zenith


class TestIrfTools(unittest.TestCase):
    def setUp(self):
        self.table = pd.DataFrame(
            {
                "dir_x": [0.1, 0.2, 0.3],
                "dir_y": [0.4, 0.5, 0.6],
                "dir_z": [0.7, 0.8, 0.9],
                "dir_x_mc": [1.1, 1.2, 1.3],
                "dir_y_mc": [0.4, 0.5, 0.6],
                "dir_z_mc": [3.7, 0.8, 2.9],
            }
        )
        self.e_bins = np.array([1, 2, 3, 4])
        self.t_bins = np.array([0, 1, 2])
        self.energy_bins = np.array([0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3])
        self.theta_bins = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
        self.w2 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1])
        self.E = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0])
        self.gamma = 2.0
        self.nevents = 1000

    def test_calc_theta_with_mc_true(self):
        expected_theta = np.array([2.83521, 2.122451, 2.682983])
        theta = irf_tools.calc_theta(self.table, mc=True)
        np.testing.assert_allclose(theta, expected_theta, atol=1e-6)

    def test_calc_theta_with_mc_false(self):
        expected_theta = np.array([2.609289, 2.549118, 2.50107])
        theta = irf_tools.calc_theta(self.table, mc=False)
        np.testing.assert_allclose(theta, expected_theta, atol=1e-6)

    def test_fill_aeff_2D(self):
        expected_aeff = np.array(
            [
                [2.195691e-12, 2.638285e-13],
                [3.293536e-12, 6.331883e-13],
                [4.391382e-12, 1.055314e-12],
            ]
        )
        aeff = irf_tools.fill_aeff_2D(
            self.e_bins,
            self.t_bins,
            self.energy_bins,
            self.theta_bins,
            self.w2,
            self.E,
            self.gamma,
            self.nevents,
        )
        np.testing.assert_allclose(aeff, expected_aeff, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
