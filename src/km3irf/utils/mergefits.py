"""Merge separated fits files into one irf fits file, which can be used in gammapy
"""

from os import path
data_dir = path.join(path.dirname(__file__), 'data')

def mergefits(aeff_fits=path.join(data_dir, "aeff.fits"), 
               psf_fits=path.join(data_dir, "psf.fits"), 
               edisp_fits=path.join(data_dir, "edisp.fits"),
               bkg_fits=path.join(data_dir, "bkg_nu.fits"),
               output_file='all_in_one.fits'):

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

def main():
    # from docopt import docopt

    # arguments = docopt(__doc__)

    # mergefits(arguments["FILE"], arguments["--raw"])
    mergefits()

if __name__ == "__main__":
    main()
