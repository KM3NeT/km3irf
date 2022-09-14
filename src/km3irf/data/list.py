from glob import glob
from os.path import getsize
from prettytable import PrettyTable
from importlib_resources import files
from os import path

#tab = PrettyTable(["File Path","Size, KB"], align="l")


def list_data():
    tab = PrettyTable(["File Path","Size, KB"], align="l")
    data_path = path.join(f"{files('km3irf')}","data","*.fits")
    # for file in glob(f"{files('km3irf')}/data/*.fits", recursive=True):
    for file in glob(data_path, recursive=True):
        #add row with file name and size in KB
        tab.add_row([file, round(getsize(filename=file)/float(1<<10), 2)])

    print(tab)

# list_data()

