from pkg_resources import get_distribution, DistributionNotFound

version = get_distribution(__name__).version
# :noindex:
__version__ = get_distribution(__name__).version

from .build_irf import DataContainer, WriteAeff, WritePSF, WriteEdisp
from .utils import *
from .test_utils import mpl_plot_check
