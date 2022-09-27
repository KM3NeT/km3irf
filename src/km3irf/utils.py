#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from .calc import Calculator
from astropy.io import fits
from os import path
from glob import glob
from os.path import getsize
from prettytable import PrettyTable
from importlib_resources import files
data_dir = path.join(path.dirname(__file__), 'data')


def print_multiplication_table(base):
    """Prints the multiplication table for a given base"""
    calculator = Calculator()
    for i in range(1, 11):
        print("{} x {} = {}".format(base, i, calculator.multiply(base, i)))


def merge_fits(aeff_fits=path.join(data_dir, "aeff.fits"), 
               psf_fits=path.join(data_dir, "psf.fits"), 
               edisp_fits=path.join(data_dir, "edisp.fits"),
               bkg_fits=path.join(data_dir, "bkg_nu.fits"),
               output_file='all_in_one.fits'):
    """Merge separated fits files into one, which can be used in gammapy"""
    hdu_list = []
    hdu_list.append(fits.PrimaryHDU())

    file_aeff = fits.open(aeff_fits) 
    hdu_list.append(file_aeff[1])
    hdu_list[1].name = 'EFFECTIVE AREA'

    file_psf = fits.open(psf_fits)
    hdu_list.append(file_psf[1])
    hdu_list[2].name = 'POINT SPREAD FUNCTION'

    file_edisp = fits.open(edisp_fits)
    hdu_list.append(file_edisp[1])
    hdu_list[3].name = 'ENERGY DISPERSION'

    file_bkg = fits.open(bkg_fits)
    hdu_list.append(file_bkg[1])
    hdu_list[4].name = 'BACKGROUND'
    
    new_fits_file = fits.HDUList(hdu_list)
    new_fits_file.writeto(path.join(data_dir, output_file), overwrite=True)

    file_aeff.close()
    file_psf.close()
    file_edisp.close()
    file_bkg.close()

def list_data():
    """Prints the table with content of data folder"""
    tab = PrettyTable(["File Path","Size, KB"], align="l")
    # data_path = path.join(f"{files('km3irf')}","data","*.fits")
    data_path = path.join(data_dir,"*.fits")
    # for file in glob(f"{files('km3irf')}/data/*.fits", recursive=True):
    for file in glob(data_path, recursive=True):
        #add row with file name and size in KB
        tab.add_row([file, round(getsize(filename=file)/float(1<<10), 2)])

    print(tab)

# merge_fits()