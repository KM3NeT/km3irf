#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from cmath import inf
from .calc import Calculator
from astropy.io import fits
from os import path, listdir
from glob import glob
from os.path import getsize
from prettytable import PrettyTable
from importlib_resources import files

data_dir = path.join(path.dirname(__file__), "data")


def print_multiplication_table(base):
    """Prints the multiplication table for a given base"""
    calculator = Calculator()
    for i in range(1, 11):
        print("{} x {} = {}".format(base, i, calculator.multiply(base, i)))


def merge_fits(
    aeff_fits=path.join(data_dir, "aeff.fits"),
    psf_fits=path.join(data_dir, "psf.fits"),
    edisp_fits=path.join(data_dir, "edisp.fits"),
    bkg_fits=path.join(data_dir, "bkg_nu.fits"),
    output_file="all_in_one.fits",
):
    """
    Merge separated fits files into one, which can be used in gammapy

    aeff_fits: path to Aeff .fits file

    psf_fits: path  to PSF .fits file

    edisp_fits: path to Edisp .fits file

    bkg_fits: path to Background .fits file

    output_file: name of the merged .fits file in data foledr of the package
    """
    hdu_list = []
    hdu_list.append(fits.PrimaryHDU())

    file_aeff = fits.open(aeff_fits)
    hdu_list.append(file_aeff[1])
    hdu_list[1].name = "EFFECTIVE AREA"

    file_psf = fits.open(psf_fits)
    hdu_list.append(file_psf[1])
    hdu_list[2].name = "POINT SPREAD FUNCTION"

    file_edisp = fits.open(edisp_fits)
    hdu_list.append(file_edisp[1])
    hdu_list[3].name = "ENERGY DISPERSION"

    file_bkg = fits.open(bkg_fits)
    hdu_list.append(file_bkg[1])
    hdu_list[4].name = "BACKGROUND"

    new_fits_file = fits.HDUList(hdu_list)
    new_fits_file.writeto(path.join(data_dir, output_file), overwrite=True)

    file_aeff.close()
    file_psf.close()
    file_edisp.close()
    file_bkg.close()


def list_data(print_tab=False):
    """
    Return dictionary with names and pathes in the data folder

    print_tab: print a table with content, default False
    """
    tab = PrettyTable(["File Path", "Size, KB"], align="l")
    data_path = path.join(data_dir, "*.fits")
    info = {}

    for file, i in zip(glob(data_path, recursive=True), listdir(data_dir)):
        tab.add_row([file, round(getsize(filename=file) / float(1 << 10), 2)])
        info.update({i: file})
    # show something
    if print_tab:
        print(tab)

    return info
