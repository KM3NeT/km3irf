# core of this code was copied from gammapy v0.17
import numpy as np
from astropy import units as u
from astropy.coordinates import Angle
from astropy.io import fits
from astropy.table import Table
from astropy.utils import lazyproperty
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from .utils import data_dir
from os import path


from .interpolation import ScaledRegularGridInterpolator
from .psf_table import EnergyDependentTablePSF, TablePSF

__all__ = ["PSF3D"]


class PSF3D:
    """PSF with axes: energy, offset, rad.

    Data format specification: :ref:`gadf:psf_table`

    Parameters
    ----------
    energy_lo : `~astropy.units.Quantity`
        Energy bins lower edges (1-dim)
    energy_hi : `~astropy.units.Quantity`
        Energy bins upper edges (1-dim)
    theta_lo : `~astropy.coordinates.Angle`
        Zenith angle bins lower edges(1-dim)
    theta_hi : `~astropy.coordinates.Angle`
        Zenith angle bins upper edges(1-dim)
    rad_lo : `~astropy.coordinates.Angle`
        Offset angle bins lower edges
    rad_hi : `~astropy.coordinates.Angle`
        Offset angle bins upper edges
    psf_value : `~astropy.units.Quantity`
        PSF (3-dim with axes: psf[rad_index, offset_index, energy_index]
    """

    def __init__(
        self,
        energy_lo,
        energy_hi,
        theta_lo,
        theta_hi,
        rad_lo,
        rad_hi,
        psf_value,
        interp_kwargs=None,
    ):
        self.energy_lo = energy_lo.to("GeV")
        self.energy_hi = energy_hi.to("GeV")
        self.theta_lo = Angle(theta_lo)
        self.theta_hi = Angle(theta_hi)
        self.rad_lo = Angle(rad_lo)
        self.rad_hi = Angle(rad_hi)
        self.psf_value = psf_value.to("sr^-1")

        self._interp_kwargs = interp_kwargs or {}

    @lazyproperty
    def _interpolate(self):
        energy = self._energy_logcenter()
        theta = self._theta_center().to("deg")
        rad = self._rad_center()

        return ScaledRegularGridInterpolator(
            points=(rad, theta, energy), values=self.psf_value, **self._interp_kwargs
        )

    def _theta_center(self):
        """Get edges of theta bins

        Returns
        -------
        theta : float
            Center of theta bins in radians.
        """
        cos_theta = (np.cos(self.theta_hi) + np.cos(self.theta_lo)) / 2.0
        return np.arccos(cos_theta)

    def _energy_logcenter(self):
        """Get logcenters of energy bins.

        Returns
        -------
        energies : float
            Logcenters of energy bins
        """
        return np.sqrt(self.energy_lo * self.energy_hi)

    def _rad_center(self):
        """Get centers of rad bins

        Returns
        -------
        rad : `~astropy.coordinates.Angle`
            rad in deg
        """
        return ((self.rad_hi + self.rad_lo) / 2.0).to("deg")

    @classmethod
    def read(cls, path_to_file=None, hdu="PSF_2D_TABLE"):
        """Create `PSF3D` from FITS file.

        Parameters
        ----------
        path_to_file : str
            File name with .fits
        hdu : str
            HDU name
        """
        if path_to_file is None:
            path_to_file = path.join(data_dir, "psf.fits")

        table = Table.read(path_to_file, hdu=hdu)
        return cls.from_table(table)

    @classmethod
    def from_table(cls, table):
        """Create `PSF3D` from `~astropy.table.Table`.

        Parameters
        ----------
        table : `~astropy.table.Table`
            Table Table-PSF info.
        """
        theta_lo = table["THETA_LO"].quantity[0]
        theta_hi = table["THETA_HI"].quantity[0]
        # offset = (theta_hi + theta_lo) / 2
        # offset = Angle(offset, unit=table["THETA_LO"].unit)

        energy_lo = table["ENERG_LO"].quantity[0]
        energy_hi = table["ENERG_HI"].quantity[0]

        rad_lo = table["RAD_LO"].quantity[0]
        rad_hi = table["RAD_HI"].quantity[0]

        psf_value = table["RPSF"].quantity[0]

        opts = {}

        return cls(
            energy_lo, energy_hi, theta_lo, theta_hi, rad_lo, rad_hi, psf_value, **opts
        )

    def to_fits(self):
        """
        Convert PSF table data to FITS HDU list.

        Returns
        -------
        hdu_list : `~astropy.io.fits.HDUList`
            PSF in HDU list format.
        """
        # Set up data
        names = [
            "ENERG_LO",
            "ENERG_HI",
            "THETA_LO",
            "THETA_HI",
            "RAD_LO",
            "RAD_HI",
            "RPSF",
        ]
        units = ["GeV", "GeV", "rad", "rad", "deg", "deg", "sr^-1"]
        data = [
            self.energy_lo,
            self.energy_hi,
            self.theta_lo,
            self.theta_hi,
            self.rad_lo,
            self.rad_hi,
            self.psf_value,
        ]

        table = Table()
        for name_, data_, unit_ in zip(names, data, units):
            table[name_] = [data_]
            table[name_].unit = unit_

        hdu = fits.BinTableHDU(table)

        return fits.HDUList([fits.PrimaryHDU(), hdu])

    def write(self, filename, *args, **kwargs):
        """Write PSF to FITS file.

        Calls `~astropy.io.fits.HDUList.writeto`, forwarding all arguments.
        """
        self.to_fits().writeto(filename, *args, **kwargs)

    def evaluate(self, energy=None, theta=None, rad=None):
        """Interpolate PSF value at a given zenith and energy.

        Parameters
        ----------
        energy : `~astropy.units.Quantity`
            energy value
        theta : `~astropy.coordinates.Angle`
            Zenith angle
        rad : `~astropy.coordinates.Angle`
            Offset wrt source position

        Returns
        -------
        values : `~astropy.units.Quantity`
            Interpolated value
        """
        if energy is None:
            energy = self._energy_logcenter()
        if theta is None:
            theta = self._theta_center()
        if rad is None:
            rad = self._rad_center()

        rad = np.atleast_1d(u.Quantity(rad))
        theta = np.atleast_1d(u.Quantity(theta))
        energy = np.atleast_1d(u.Quantity(energy))
        return self._interpolate(
            (
                rad[:, np.newaxis, np.newaxis],
                theta[np.newaxis, :, np.newaxis],
                energy[np.newaxis, np.newaxis, :],
            )
        )

    def to_energy_dependent_table_psf(self, theta=None, rad=None, exposure=None):
        """
        Convert PSF3D in EnergyDependentTablePSF.

        Parameters
        ----------
        theta : `~astropy.coordinates.Angle`
            Zenith angle
        rad : `~astropy.coordinates.Angle`
            Offset from PSF center used for evaluating the PSF on a grid.
            Default is the ``rad`` from this PSF.
        exposure : `~astropy.units.Quantity`
            Energy dependent exposure. Should be in units equivalent to 'cm^2 s'.
            Default exposure = 1.

        Returns
        -------
        table_psf : `~psf_table.EnergyDependentTablePSF`
            Energy-dependent PSF
        """

        if theta is None:
            theta = self._theta_center()
        else:
            theta = Angle(theta)

        energies = self._energy_logcenter()

        if rad is None:
            rad = self._rad_center()
        else:
            rad = Angle(rad)

        psf_value = self.evaluate(theta=theta, rad=rad).squeeze()
        return EnergyDependentTablePSF(
            energy=energies, rad=rad, exposure=exposure, psf_value=psf_value.T
        )

    def to_table_psf(self, energy, theta="0 deg", **kwargs):
        """Create `~psf_table.TablePSF` at one given energy.

        Parameters
        ----------
        energy : `~astropy.units.Quantity`
            Energy
        theta : `~astropy.coordinates.Angle`
            Zenith angle. Default theta = 0 deg

        Returns
        -------
        psf : `~psf_table.TablePSF`
            Table PSF
        """
        energy = u.Quantity(energy)
        theta = Angle(theta)
        psf_value = self.evaluate(energy, theta.to("rad")).squeeze()
        rad = self._rad_center()
        return TablePSF(rad, psf_value, **kwargs)

    def containment_radius(
        self, energy, theta="0 deg", fraction=0.68, interp_kwargs=None
    ):
        """Containment radius.

        Parameters
        ----------
        energy : `~astropy.units.Quantity`
            Energy
        theta : `~astropy.coordinates.Angle`
            Zenith angle. Default theta = 0 deg
        fraction : float
            Containment fraction. Default fraction = 0.68

        Returns
        -------
        radius : `~astropy.units.Quantity`
            Containment radius in deg
        """
        energy = np.atleast_1d(u.Quantity(energy))
        theta = np.atleast_1d(u.Quantity(theta))

        radii = []
        for t in theta:
            psf = self.to_energy_dependent_table_psf(theta=t)
            radii.append(psf.containment_radius(energy, fraction=fraction))

        return u.Quantity(radii).T.squeeze()

    def plot_containment_vs_energy(
        self, fractions=[0.68, 0.90], thetas=[90, 120], ax=None, **kwargs
    ):
        """Plot containment fraction as a function of energy."""

        ax = plt.gca() if ax is None else ax

        n_lines = len(fractions) * len(thetas)
        cmap = cm.plasma
        line_colors = []
        for i in range(n_lines):
            line_colors.append(cmap(i / n_lines))

        thetas = Angle(thetas, "deg")

        label_fontsize = kwargs.pop("label_fontsize", 14)
        tick_fontsize = kwargs.pop("tick_fontsize", 12)

        energy = (
            np.logspace(
                np.log10(self.energy_lo[0].value),
                np.log10(self.energy_hi[-1].value),
                101,
            )
            * self.energy_lo.unit
        )

        count = 0
        for theta in thetas:
            for fraction in fractions:
                radius = self.containment_radius(energy, theta, fraction)
                label = f"{theta.deg} deg, {100 * fraction:.1f}%"
                ax.plot(
                    energy.value,
                    radius.value,
                    label=label,
                    color=line_colors[count],
                )
                ax.legend(fontsize=tick_fontsize)
                count += 1

        ax.semilogx()
        ax.legend(loc="best")
        ax.set_xlabel(
            f"Energy, [{str(self.energy_lo[0].unit)}]", fontsize=label_fontsize
        )
        ax.set_ylabel("Containment radius, [deg]", fontsize=label_fontsize)
        ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)

        return ax

    def plot_psf_vs_rad(self, theta="0 deg", energy=u.Quantity(10.0, "GeV"), **kwargs):
        """Plot PSF vs rad.

        Parameters
        ----------
        energy : `~astropy.units.Quantity`
            Energy. Default energy = 10 GeV
        theta : `~astropy.coordinates.Angle`
            Zenith angle. Default theta = 0 deg

        Returns
        -------
        plot : `~psf_table.plot_psf_vs_rad`
        """

        theta = Angle(theta)
        table = self.to_table_psf(theta=theta, energy=energy)
        return table.plot_psf_vs_rad(energy=energy, **kwargs)

    def plot_containment(
        self, fraction=0.68, ax=None, show_safe_energy=False, add_cbar=True, **kwargs
    ):
        """
        Plot containment image with energy and theta axes.

        Parameters
        ----------
        fraction : float
            Containment fraction between 0 and 1.
        add_cbar : bool
            Add a colorbar
        """
        import matplotlib.pyplot as plt

        ax = plt.gca() if ax is None else ax

        energy = self._energy_logcenter()
        theta = self._theta_center()

        # Set up and compute data
        containment = self.containment_radius(energy, theta, fraction)

        # plotting defaults
        kwargs.setdefault("cmap", "RdPu")
        # kwargs.setdefault("vmin", np.nanmin(containment.value))
        # kwargs.setdefault("vmax", np.nanmax(containment.value))
        kwargs.setdefault(
            "norm",
            Normalize(
                vmin=np.nanmin(containment.value), vmax=np.nanmax(containment.value)
            ),
        )
        label_fontsize = kwargs.pop("label_fontsize", 14)
        tick_fontsize = kwargs.pop("tick_fontsize", 12)

        # Plotting
        x = energy.value
        y = theta.to("deg").value
        caxes = ax.pcolormesh(x, y, containment.value.T, shading="auto", **kwargs)

        # Axes labels and ticks, colobar
        ax.semilogx()
        ax.set_ylabel("Theta, [deg]", fontsize=label_fontsize)
        ax.set_xlabel(f"Energy, [{energy.unit}]", fontsize=label_fontsize)
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.min(), y.max())

        # Set font size for axis ticks
        ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)

        if show_safe_energy:
            self._plot_safe_energy_range(ax)

        if add_cbar:
            label = f"Containment radius R{100 * fraction:.0f}, [{containment.unit}]"
            cbar = ax.figure.colorbar(caxes, ax=ax)
            cbar.ax.tick_params(labelsize=tick_fontsize)
            cbar.set_label(label=label, fontsize=label_fontsize)

        return ax

    def _plot_safe_energy_range(self, ax):
        """add safe energy range lines to the plot"""
        esafe = self.energy_thresh_lo
        omin = self.offset.value.min()
        omax = self.offset.value.max()
        ax.hlines(y=esafe.value, xmin=omin, xmax=omax)
        label = f"Safe energy threshold: {esafe:3.2f}"
        ax.text(x=0.1, y=0.9 * esafe.value, s=label, va="top")

    def peek(self, figsize=(15, 4), **kwargs):
        """Quick-look summary plots for PSF.

        Parameters
        ----------
        figsize : tuple
            Size of the figure.
        """
        import matplotlib.pyplot as plt

        label_fontsize = kwargs.pop("label_fontsize", 12)
        tick_fontsize = kwargs.pop("tick_fontsize", 10)

        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=figsize)

        self.plot_psf_vs_rad(ax=axes[0])
        self.plot_containment(fraction=0.90, ax=axes[1])
        self.plot_containment_vs_energy(ax=axes[2])

        plt.tight_layout()
