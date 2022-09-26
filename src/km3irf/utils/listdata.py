"""
Show the content of the data folder including pathes to files
"""

from glob import glob
from os.path import getsize
from prettytable import PrettyTable
from importlib_resources import files
from os import path



def listdata():
    tab = PrettyTable(["File Path","Size, KB"], align="l")
    data_path = path.join(f"{files('km3irf')}","data","*.fits")
    # for file in glob(f"{files('km3irf')}/data/*.fits", recursive=True):
    for file in glob(data_path, recursive=True):
        #add row with file name and size in KB
        tab.add_row([file, round(getsize(filename=file)/float(1<<10), 2)])

    print(tab)



def main():
    # from docopt import docopt

    # arguments = docopt(__doc__)

    # h5info(arguments["FILE"], arguments["--raw"])
    listdata()


if __name__ == "__main__":
    main()
