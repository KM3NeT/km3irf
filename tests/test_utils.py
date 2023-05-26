#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest
from os import path, listdir, curdir, remove
import uproot as ur
from astropy.io import fits
from km3net_testdata import data_path

from km3irf import utils
from km3irf.test_utils import mpl_plot_check


class TestUtils(unittest.TestCase):
    def setUp(self):
        self.test_path = path.join(path.abspath(curdir), "src", "km3irf", "data")

    def test_merge_fits(self):
        utils.merge_fits()
        assert "all_in_one.fits" in listdir(self.test_path)

    def test_list_data(self):
        numb_fits = [i for i in listdir(self.test_path) if ".fits" in i]
        assert len(utils.list_data()) == len(numb_fits)


class TestDrawClasses(unittest.TestCase):
    def setUp(self):
        self.aeff = utils.DrawAeff()

    def test_init(self):
        assert len(self.aeff.energy_center) == 48
        assert len(self.aeff.zenith) == 12

    def test_plot(self):
        with mpl_plot_check():
            self.aeff.plot_aeff()

        with mpl_plot_check():
            self.aeff.plot_energy_dependence()

        with mpl_plot_check():
            self.aeff.plot_zenith_dependence()


if __name__ == "__main__":
    unittest.main()
