#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest
from os import path, listdir, curdir, remove
import uproot as ur
from astropy.io import fits
from km3net_testdata import data_path

from km3irf import build_irf


class TestBuild_IRF(unittest.TestCase):
    def setUp(self):
        self.testdata = data_path("dst/mcv5.1.km3_numuCC.ALL.dst.bdt.10events.root")
        self.init_data = build_irf.DataContainer(no_bdt=False, infile=self.testdata)

    def test_apply_cuts(self):
        self.init_data.apply_cuts
        assert self.init_data.df.shape[0] != None

    def test_unpack_data(self):
        df_test = build_irf.unpack_data(
            no_bdt=False, uproot_file=ur.open(self.testdata)
        )
        assert (df_test.size == 110) | (df_test.size == 90)

    def test_buid_aeff(self):
        self.init_data.build_aeff()
        size_of = path.getsize(path.join(path.abspath(curdir), "aeff.fits"))
        with fits.open(path.join(path.abspath(curdir), "aeff.fits")) as file_fits:
            global header_fits
            header_fits = file_fits[1].header["EXTNAME"]
        assert "aeff.fits" in listdir(path.abspath(curdir))
        assert size_of != 0
        assert header_fits == "EFFECTIVE AREA"
        remove(path.join(path.abspath(curdir), "aeff.fits"))

    def test_buid_psf(self):
        self.init_data.build_psf()
        size_of = path.getsize(path.join(path.abspath(curdir), "psf.fits"))
        with fits.open(path.join(path.abspath(curdir), "psf.fits")) as file_fits:
            global header_fits
            header_fits = file_fits[1].header["EXTNAME"]
        assert "psf.fits" in listdir(path.abspath(curdir))
        assert size_of != 0
        assert header_fits == "PSF_2D_TABLE"
        remove(path.join(path.abspath(curdir), "psf.fits"))

    def test_buid_edisp(self):
        self.init_data.build_edisp()
        self.file_name = "edisp.fits"
        size_of = path.getsize(path.join(path.abspath(curdir), self.file_name))
        with fits.open(path.join(path.abspath(curdir), self.file_name)) as file_fits:
            global header_fits
            header_fits = file_fits[1].header["EXTNAME"]
        assert self.file_name in listdir(path.abspath(curdir))
        assert size_of != 0
        assert header_fits == "EDISP_2D"
        remove(path.join(path.abspath(curdir), self.file_name))


if __name__ == "__main__":
    unittest.main()
