"""Functions and classes for calculation of ground motion parameters."""

import re

import numpy as np
import numpy.typing as npt

from . import FPATH_DATA
from .utils import check_bounds, check_options, dist_lognorm

_CACHE_REA15 = {}


def calc_damping_scaling_rea15(
    damping: float,
    mag: float,
    dist_rup: float,
    comp: str = "rotd50",
    periods: npt.ArrayLike | None = None,
) -> npt.ArrayLike:
    """Compute the damping scaling proposed by Rezaeian et al. (2014).

    Parameters
    ----------
    damping: damping ratio of the oscillator in percent (2 for 2%)
    mag: earthquake magnitude
    dist_rup: closest distance to the rupture in [km]
    comp: component, can be either: 'rotd50', 'roti50', 'vertical'
    periods: *optional* periods to provide values.

    Returns
    -------
    ret : np.rec.array
        Array with columns period, damping scaling factor, and natural logarithmic standard deviation.
    """
    check_bounds(damping, 0.5, 30, "damping")
    check_options(comp, ["rotd50", "roti50", "vertical"], "comp")

    # Only load the coefficients once
    if comp not in _CACHE_REA15:
        _CACHE_REA15[comp] = np.genfromtxt(
            FPATH_DATA / f"rezaeian_et_al_2014-{comp}.csv",
            delimiter=",",
            skip_header=1,
            names=True,
        ).view(np.recarray)
    C = _CACHE_REA15[comp]

    ln_damp = np.log(damping)
    ln_dsf = (
        C.b0
        + C.b1 * ln_damp
        + C.b2 * ln_damp**2
        + (C.b3 + C.b4 * ln_damp + C.b5 * ln_damp**2) * mag
        + (C.b6 + C.b7 * ln_damp + C.b8 * ln_damp**2) * np.log(dist_rup + 1)
    )
    ln_damp_5 = np.log(damping / 5)
    ln_std = np.abs(C.a0 * ln_damp_5 + C.a1 * ln_damp_5**2)

    if periods:
        # Interpolate over the provided periods
        ln_dsf = np.interp(np.log(periods), np.log(C.period), ln_dsf)
        ln_std = np.interp(np.log(periods), np.log(C.period), ln_std)
    else:
        periods = C.period

    ret = np.rec.fromarrays(
        [periods, np.exp(ln_dsf), ln_std], names="period,dsf,ln_std"
    )
    return ret


@dist_lognorm
def calc_period_rea05(
    kind: str,
    mag: float,
    dist_rup: float,
    site_class: str = "c",
    directivity: bool = False,
    **kwargs,
):
    """Rathje et al. (2005) period metrics."""
    # Model coefficients from Table 2
    C = {
        model: np.rec.fromrecords(values, names="c1,c2,c3,c4,c5,c6")
        for model, values in zip(
            ["period_mean", "period_avg", "period_pred"],
            [
                (-1.00, 0.18, 0.0038, 0.078, 0.27, 0.40),
                (-0.89, 0.29, 0.0030, 0.070, 0.25, 0.37),
                (-1.78, 0.30, 0.0045, 0.150, 0.33, 0.24),
            ],
        )
    }[kind]

    check_options(site_class, "bcd", "site_class")
    if kind in ["period_avg", "period_pred"]:
        check_bounds(mag, 4.7, 7.6, "mag")
    else:
        mag = np.minimum(mag, 7.25)
        check_bounds(mag, 5.0, 7.25, "mag")

    if site_class == "c":
        s_c = 1
        s_d = 0
    elif site_class == "d":
        s_c = 0
        s_d = 1
    else:
        s_c = 0
        s_d = 0

    f_d = int(directivity)

    ln_period = (
        C.c1
        + C.c2 * (mag - 6)
        + C.c3 * dist_rup
        + C.c4 * s_c
        + C.c5 * s_d
        + C.c6 * (1 - dist_rup / 20) * f_d
    )

    intra = {
        "period_mean": {"b": 0.42, "c": 0.38, "d": 0.31},
        "period_avg": {"b": 0.42, "c": 0.38, "d": 0.31},
        "period_pred": {"b": 0.42, "c": 0.38, "d": 0.31},
    }[kind][site_class]
    inter = {"period_mean": 0.17, "period_avg": 0.13, "period_pred": 0.22}[kind]
    ln_std = np.sqrt(intra**2 + inter**2)

    return ln_period, ln_std


@dist_lognorm
def calc_aris_intensity_aea16(
    mag: float,
    v_s30: float,
    pga: float,
    psa_1s: float,
    hanging_wall: bool = False,
    dist_jb: float | None = None,
    ln_std_pga: float | None = None,
    ln_std_psa_1s: float | None = None,
    **kwargs,
):
    """Arias intensity estimate from Abrahamson, Shi, and Yang (2016)."""
    # From Table 3.1
    # Value of c8 is provided after equation 3.7
    C = np.rec.fromrecords(
        (0.47, -0.28, 0.50, 1.52, 0.21, 0.09), names="c1,c2,c3,c4,c5,c8"
    )

    ln_arias_int = (
        C.c1
        + C.c2 * np.log(v_s30)
        + C.c3 * mag
        + C.c4 * np.log(pga)
        + C.c5 * np.log(psa_1s)
    )
    if hanging_wall:
        assert dist_jb is not None
        ln_arias_int += C.c8 * np.clip(1 - (dist_jb - 5) / 5, 0, 1)

    if ln_std_pga is None and ln_std_psa_1s is None:
        # Computed from residuals
        ln_std = np.interp(
            mag, [3, 4, 5, 6, 7], [0.40, 0.39, 0.37, 0.33, 0.40], left=0.40, right=0.40
        )
    else:
        raise NotImplementedError

    return ln_arias_int, ln_std


@dist_lognorm
def calc_duration_afs16(
    kind: str,
    mag: float,
    dist_rup: float,
    v_s30: float,
    mech: str = "strike-slip",
    z1: float | None = None,
    **kwargs,
):
    """Significant duration from Afshari and Stewart (2016).

    Parameters
    ----------
    kind: duration parameter, one of 'd5-75', 'd5-95', 'd20-80', the time
        intervals over which 5-75%, 5-95%, or 20-80% of the Arias intensity
        accumulates.
    mag: moment magnitude.
    dist_rup: closest distance to the rupture in [km].
    v_s30: time-averaged shear-wave velocity in the top 30 m in [m/s].
    mech: focal mechanism, one of 'normal', 'reverse', 'strike-slip',
        'unknown'.
    z1: depth to the 1.0 km/s shear-wave isosurface in [m]. The basin-depth
        term of the published model is not implemented in this version, so
        supplying a value raises NotImplementedError rather than silently
        ignoring it.

    Returns
    -------
    ret : float
        Median significant duration in [sec], or (ln median, ln standard
        deviation) when ``stats=True`` is passed.

    Notes
    -----
    The model is additive in arithmetic units: the median duration is
    ``(source + path) * exp(site)``, where the source duration is the
    inverse corner frequency from the Brune model with a magnitude-dependent
    stress-drop index, the path duration is piecewise linear in rupture
    distance with breaks at 10 and 50 km, and the site term scales with
    ``v_s30`` up to a limiting velocity of 600 m/s. Coefficients are from
    Tables 1-3 of Afshari and Stewart (2016). The basin-depth term
    (their Eq. 9) is omitted; see the ``z1`` parameter note above.
    """
    if z1 is not None:
        raise NotImplementedError(
            "The basin-depth term of Afshari and Stewart (2016) is not "
            "implemented; pass z1=None to use the v_s30 site term only."
        )
    check_options(kind, ["d5-75", "d5-95", "d20-80"], "kind")
    check_options(mech, ["normal", "reverse", "strike-slip", "unknown"], "mech")
    check_bounds(mag, 3.0, 8.0, "mag")
    check_bounds(dist_rup, 0, 300, "dist_rup")
    check_bounds(v_s30, 150, 1500, "v_s30")

    # Coefficients from Tables 1-3 of Afshari and Stewart (2016).
    # M1, M2: magnitude breaks; b0: source duration at M1 in [sec];
    # b1: stress-drop index at M* = 6; b2, b3: stress-drop slopes below
    # and above M2; c1-c3: path slopes in [sec/km]; c4: v_s30 slope;
    # v_ref: reference velocity in [m/s]; tau, phi: between- and
    # within-event standard deviations at low and high magnitude.
    C = {
        "d5-75": {
            "m1": 5.35,
            "m2": 7.15,
            "b0": {
                "normal": 1.603,
                "reverse": 0.7806,
                "strike-slip": 1.279,
                "unknown": 1.280,
            },
            "b1": {
                "normal": 4.992,
                "reverse": 7.061,
                "strike-slip": 5.578,
                "unknown": 5.576,
            },
            "b2": 0.9011,
            "b3": -1.684,
            "c1": 0.1159,
            "c2": 0.1065,
            "c3": 0.0682,
            "c4": -0.2246,
            "v_ref": 368.2,
            "tau": (0.31, 0.25),
            "phi": (0.54, 0.41),
        },
        "d5-95": {
            "m1": 5.2,
            "m2": 7.4,
            "b0": {
                "normal": 2.541,
                "reverse": 1.612,
                "strike-slip": 2.302,
                "unknown": 2.182,
            },
            "b1": {
                "normal": 3.170,
                "reverse": 4.536,
                "strike-slip": 3.467,
                "unknown": 3.628,
            },
            "b2": 0.9443,
            "b3": -3.911,
            "c1": 0.3165,
            "c2": 0.2539,
            "c3": 0.0932,
            "c4": -0.3183,
            "v_ref": 369.9,
            "tau": (0.24, 0.23),
            "phi": (0.47, 0.35),
        },
        "d20-80": {
            "m1": 5.2,
            "m2": 7.4,
            "b0": {
                "normal": 1.413,
                "reverse": 0.7754,
                "strike-slip": 0.8833,
                "unknown": 0.8851,
            },
            "b1": {
                "normal": 4.778,
                "reverse": 6.579,
                "strike-slip": 6.188,
                "unknown": 6.182,
            },
            "b2": 0.7414,
            "b3": -3.164,
            "c1": 0.0646,
            "c2": 0.0865,
            "c3": 0.0373,
            "c4": -0.4237,
            "v_ref": 369.6,
            "tau": (0.30, 0.22),
            "phi": (0.62, 0.45),
        },
    }[kind]

    # Source duration: constant below M1, otherwise the inverse corner
    # frequency for a Brune source with shear-wave velocity 3.2 km/s.
    if mag <= C["m1"]:
        source = C["b0"][mech]
    else:
        m0 = 10 ** (1.5 * mag + 16.05)
        ln_stress = C["b1"][mech] + C["b2"] * (min(mag, C["m2"]) - 6.0)
        if mag > C["m2"]:
            ln_stress += C["b3"] * (mag - C["m2"])
        corner_freq = 4.9e6 * 3.2 * (np.exp(ln_stress) / m0) ** (1 / 3)
        source = 1 / corner_freq

    # Path duration: piecewise linear, breaks at 10 and 50 km.
    if dist_rup <= 10:
        path = C["c1"] * dist_rup
    elif dist_rup <= 50:
        path = C["c1"] * 10 + C["c2"] * (dist_rup - 10)
    else:
        path = C["c1"] * 10 + C["c2"] * 40 + C["c3"] * (dist_rup - 50)

    # Site term: v_s30 scaling only, capped at 600 m/s.
    site = C["c4"] * np.log(min(v_s30, 600) / C["v_ref"])

    ln_mean = np.log(source + path) + site

    # Standard deviation: between-event term interpolates over
    # 6.25-6.5, within-event over 5.5-5.75.
    tau1, tau2 = C["tau"]
    phi1, phi2 = C["phi"]
    tau = (
        tau1
        if mag <= 6.25
        else tau2 if mag >= 6.5 else tau1 + (tau2 - tau1) * (mag - 6.25) / 0.25
    )
    phi = (
        phi1
        if mag <= 5.5
        else phi2 if mag >= 5.75 else phi1 + (phi2 - phi1) * (mag - 5.5) / 0.25
    )
    ln_std = np.sqrt(tau**2 + phi**2)

    return ln_mean, ln_std


class TimeSeries:
    def __init__(self, time_step, accels, info=""):
        self._time_step = time_step
        self._accels = accels
        self._info = info

    @property
    def info(self):
        return self._info

    @property
    def time_step(self):
        return self._time_step

    @property
    def accels(self):
        return self._accels

    @classmethod
    def read_at2(cls, filename):
        with open(filename) as fp:
            next(fp)
            info = next(fp).strip()
            next(fp)
            parts = [p for p in re.split("[ ,]", next(fp)) if p]
            _ = int(parts[1])
            time_step = float(parts[3])
            accels = np.array([p for line in fp for p in line.split()]).astype(float)
        return cls(time_step, accels, info)


# FIXME: Add
# @dist_lognorm
# def calc_conditional_pgv_ab19(
#         mag,
#         dist,
#         v_s30,
#
#         **kwds
# ):
#     """Compute the PGV/PGA ratio from Abrahamson and Bhasin (2018).
#
#     """
#
#     if pga is not None:
#         c_values = ()
#     if psa_1s is not None:
#
#     else:
#         C = np.rec.fromrecords(
#             ()
#         )
#
#
#     ln_mean = (
#         3.3 +
#         0.53 * mag -
#         0.14 * np.log(dist + 3) -
#         0.32 * np.log(v_s30) - np.log(980)
#     )
#     ln_std = np.sqrt(0.45 ** 2 + (0.53 * 0.3) ** 2) * np.ones_like(mag)
#     return ln_mean, ln_std


def calc_pulse_proportion_hea12(
    dist_rup: npt.ArrayLike, epsilon: npt.ArrayLike
) -> npt.ArrayLike:
    """Calculate the pulse proportion from Hayden et al. (2012).

    Parameters
    ----------
    dist_rup : float or array_like
        Closest distance to the fault rupture [km]
    epsilon : float or array_like
        Number of stanard deviations

    Returns
    -------
    Proportion of records with pulse-like behavior

    """

    prop = np.exp(0.891 - 0.188 * dist_rup + 1.230 * epsilon) / (
        1 + np.exp(0.981 - 0.188 * dist_rup + 1.230 * epsilon)
    )

    return prop


@dist_lognorm
def calc_site_atten_vhea11(
    v_s30: float | npt.ArrayLike, **kwargs
) -> float | np.ndarray:
    r"""Site attenuation from Van Houtte et al. (2011).

    Parameters
    ----------
    v_s30 : float | ArrayLike
        time-average velocity in the top 30 m in m/s.
    **kwargs
        Passed to `dist_lognorm`

    Returns
    -------
    float | np.ndarray
        Predicted site attenuation (\kappa_0) in sec
    """
    ln_mean = 3.49 - 1.062 * np.log(v_s30)
    ln_std = 0.55
    return ln_mean, ln_std


@dist_lognorm
def calc_site_atten_xr21(v_s30: float | npt.ArrayLike, **kwargs) -> float | np.ndarray:
    r"""Site attenuation from Xu and Rathje (2021).

    Parameters
    ----------
    v_s30 : float | ArrayLike
        time-average velocity in the top 30 m in m/s.
    **kwargs
        Passed to `dist_lognorm`

    Returns
    -------
    float | np.ndarray
        Predicted site attenuation (\kappa_0) in sec

    """
    ln_vs30 = np.log(np.clip(v_s30, 155, 2000))
    ln_mean = -0.18 * ln_vs30**2 + 1.816 * ln_vs30 - 7.38
    ln_std = 0.3
    return ln_mean, ln_std
