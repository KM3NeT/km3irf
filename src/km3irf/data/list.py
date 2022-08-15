from glob import glob
from os.path import getsize
from prettytable import PrettyTable
from importlib_resources import files

#tab = PrettyTable(["File Path","Size, KB"], align="l")


def list_data():
    tab = PrettyTable(["File Path","Size, KB"], align="l")
    for file in glob(f"{files('km3irf')}/data/*.fits", recursive=True):
        #add row with file name and size in KB
        tab.add_row([file, round(getsize(filename=file)/float(1<<10), 2)])

    print(tab)

# list_data()

