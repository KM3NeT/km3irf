from pkg_resources import get_distribution, DistributionNotFound

version = get_distribution(__name__).version
__version__ = get_distribution(__name__).version

from .calc import Calculator
from .build_aeff import DataContainer, WriteAeff
from .utils import *
