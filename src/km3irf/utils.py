#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from .calc import Calculator
from astropy.io import fits
from os import path
data_dir = path.join(path.dirname(__file__), 'data')


def print_multiplication_table(base):
    """Prints the multiplication table for a given base"""
    calculator = Calculator()
    for i in range(1, 11):
        print("{} x {} = {}".format(base, i, calculator.multiply(base, i)))


# def merge_fits(aeff_fits="./data/aeff.fits", 
#                psf_fits="./data/psf.fits", 
#                edisp_fits="./data/edisp.fits",
#                bkg_fits="./data/bkg_nu.fits",
#                output_file='all_in_one.fits'):
def merge_fits(aeff_fits=path.join(data_dir, "aeff.fits"), 
               psf_fits=path.join(data_dir, "psf.fits"), 
               edisp_fits=path.join(data_dir, "edisp.fits"),
               bkg_fits=path.join(data_dir, "bkg_nu.fits"),
               output_file='all_in_one.fits'):
    """Merge separated fits files into one, which can be used in gammapy"""
    hdu_list = []
    hdu_list.append(fits.PrimaryHDU())

    with fits.open(aeff_fits) as file_aeff:
        hdu_list.append(file_aeff[1])
    hdu_list[1].name = 'EFFECTIVE AREA'

    with fits.open(psf_fits) as file_psf:
        hdu_list.append(file_psf[1])
    hdu_list[2].name = 'POINT SPREAD FUNCTION'

    with fits.open(edisp_fits) as file_edisp:
        hdu_list.append(file_edisp[1])
    hdu_list[3].name = 'ENERGY DISPERSION'

    with fits.open(bkg_fits) as file_bkg:
        hdu_list.append(file_bkg[1])
    hdu_list[4].name = 'BACKGROUND'

    new_fits_file = fits.HDUList(hdu)
    new_fits_file.writeto(f'./data/{output_file}', overwrite=True)

