from ase.units import Bohr, Hartree
from dataclasses import dataclass
from typing import Union
import json
import numpy as np
import scipy.optimize

from mec_sandia.gaussians import estimate_kinetic_energy_sampling
from mec_sandia.density_matrix import DensityMatrix


def compute_sigma_time(time, sigma, stopping_deriv, mass_proj):
    return sigma * (1 + abs(stopping_deriv) * time / mass_proj)


def _fit_linear(x, a, b):
    return a * x + b


class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


@dataclass
class StoppingPowerData:
    stopping: float
    stopping_err: float
    kinetic: np.ndarray
    kinetic_err: np.ndarray
    num_samples: int
    intercept: float
    intercept_err: float
    time_vals: np.ndarray
    distance: np.ndarray
    sigma_vals: np.ndarray
    stopping_expected: float

    def linear_fit(self, xs):
        return _fit_linear(xs, self.stopping, self.intercept)

    def to_file(self, filename):
        json_string = json.dumps(self.__dict__, cls=NumpyEncoder, indent=4)
        with open(filename, "w") as fid:
            fid.write(json_string)

    @staticmethod
    def from_file(filename):
        with open(filename, "r") as fid:
            stopping_data_dict = json.load(fid)
        for k, v in stopping_data_dict.items():
            if isinstance(v, list):
                stopping_data_dict[k] = np.array(v)
        return StoppingPowerData(**stopping_data_dict)


@dataclass
class DFTStoppingPowerData:
    times: np.ndarray
    position: np.ndarray
    work: np.ndarray
    kinetic: np.ndarray
    velocity: np.ndarray
    kproj: np.ndarray
    kproj_sub_sample: np.ndarray
    time_sub_sample: np.ndarray
    distance: np.ndarray
    stopping: float = 0.0


def _remove_rare_events(
    distance: np.ndarray,
    kproj_vals: np.ndarray,
    mass_proj: float,
    outlier_scale: float = 5,
):
    yvals = kproj_vals**2.0 / (2 * mass_proj)
    xvals = distance
    # pylint: disable=unbalanced-tuple-unpacking
    popt, _ = scipy.optimize.curve_fit(
        _fit_linear,
        xvals,
        yvals,
    )
    distances = np.abs(yvals - _fit_linear(xvals, popt[0], popt[1]))
    mean_dist = np.mean(distances)
    keep = np.where(distances < outlier_scale * mean_dist)
    return keep


def parse_stopping_data(
    filename: str,
    v_proj: float,
    mass_proj: int = 1836,
    num_points: int = 20,
    rare_event: float = -1,
    random_sub_sample: bool=True,
    max_time: float=-1,
    stopping_data_filename: Union[str, None] = None,
) -> DFTStoppingPowerData:
    qData1 = np.loadtxt(filename)
    position_au = qData1[:, 0] / Bohr
    time_au = position_au / v_proj
    work_au = qData1[:, 1] / Hartree
    ke_time = 0.5 * mass_proj * (v_proj**2) - work_au
    velocity_time = np.sqrt(
        2.0 / mass_proj * ke_time
    )  # 1/2 mv^2 = KE, so sqrt(2/m*KE)=v
    kproj_time = mass_proj * velocity_time
    if rare_event > 0:
        keep = _remove_rare_events(position_au, kproj_time, mass_proj, rare_event)
    else:
        keep = np.arange(len(time_au))
    # TODO: Replace this with Alina's sampling!
    time_keep = time_au[keep]
    max_time_indx = np.where(time_keep < max_time)[0][-1]
    time_keep = time_keep[:max_time_indx]
    if random_sub_sample:
        sub_sample = np.random.choice(np.arange(len(time_keep[keep])), num_points)
    else:
        len_data = len(time_keep)
        skip_every = len_data//num_points + 1
        sub_sample = np.arange(len_data)[::skip_every]
    time_vals = time_au[keep][sub_sample]
    # Sorting because sub_sample will not be ordered
    ix = np.argsort(time_vals)
    time_vals = time_vals[ix]
    kproj_x_vals = kproj_time[keep][sub_sample][ix]
    if stopping_data_filename is not None:
        qData2 = np.loadtxt(stopping_data_filename)
        _vel_au = qData2[:, 0]
        stopping = float(qData2[np.where(abs(_vel_au - v_proj) < 1e-8), 1]) * (
            Bohr / Hartree
        )
    else:
        stopping = 0.0
    data = DFTStoppingPowerData(
        time_au[keep],
        position_au[keep],
        work_au[keep],
        ke_time[keep],
        velocity_time[keep],
        kproj_time[keep],
        kproj_x_vals,
        time_vals,
        time_au[keep] * velocity_time[keep],
        stopping=stopping,
    )
    return data


def compute_stopping_power(
    ecut_hartree: float,
    box_length: float,
    sigma0: float,
    time_vals: np.ndarray,
    kproj_vals: np.ndarray,
    stopping_deriv: float,
    mass_proj: float,
    ndim: int = 3,
    num_samples: int = 10_000,
) -> StoppingPowerData:
    sigma_tvals = compute_sigma_time(time_vals, sigma0, stopping_deriv, mass_proj)
    func = lambda x, k: estimate_kinetic_energy_sampling(
        ecut_hartree, box_length, x, ndim=ndim, num_samples=num_samples, kproj=k
    )
    values = [func(sigma_t, kproj) for (sigma_t, kproj) in zip(sigma_tvals, kproj_vals)]
    yvals, errs = zip(*values)
    yvals = np.array(yvals) / mass_proj
    errs = np.array(errs) / mass_proj
    velocity_vals = kproj_vals[:, 0] / mass_proj
    distance = time_vals * velocity_vals
    # pylint: disable=unbalanced-tuple-unpacking
    popt, pcov = scipy.optimize.curve_fit(
        _fit_linear, distance, yvals, sigma=errs, absolute_sigma=True
    )
    slope, incpt = popt
    slope_err = np.sqrt(pcov[0, 0])
    yvals = (np.sum(kproj_vals**2.0, axis=-1) + sigma_tvals**2.0) / (2 * mass_proj)
    xvals = distance
    expected_val = compute_stopping_exact(xvals, yvals)
    data = StoppingPowerData(
        stopping=slope,
        stopping_err=slope_err,
        kinetic=yvals,
        kinetic_err=errs,
        num_samples=num_samples,
        intercept=incpt,
        intercept_err=np.sqrt(pcov[0, 0]),
        time_vals=time_vals,
        distance=distance,
        sigma_vals=sigma_tvals,
        stopping_expected=expected_val,
    )
    return data


def compute_stopping_power_electrons(
    eigs: np.ndarray,
    occs: np.ndarray,
    sigma0: float,
    time_vals: np.ndarray,
    kproj_vals: np.ndarray,
    stopping_deriv: np.ndarray,
    mass_proj: float,
    num_samples: int = 10_000,
) -> StoppingPowerData:
    """Slight overkill"""

    def func(kproj, k0):
        dm_1eV = DensityMatrix.build_grand_canonical(occs, num_samples)
        e1b, err = dm_1eV.contract_diagonal_one_body(eigs)
        energy_proj = np.dot(kproj, kproj) / (2 * mass_proj) - np.dot(k0, k0) / (
            2 * mass_proj
        )
        e1b += -1 * energy_proj
        return e1b, err

    values = [func(kproj, kproj_vals[0]) for kproj in kproj_vals]
    yvals, errs = zip(*values)
    yvals = np.array(yvals)
    errs = np.array(errs)
    velocity_vals = kproj_vals[:, 0] / mass_proj
    distance = time_vals * velocity_vals
    popt, pcov = scipy.optimize.curve_fit(
        _fit_linear, distance, yvals, sigma=errs, absolute_sigma=True
    )
    slope, incpt = popt
    slope_err = np.sqrt(pcov[0, 0])
    sigma_tvals = compute_sigma_time(time_vals, sigma0, stopping_deriv, mass_proj)
    yvals = (np.sum(kproj_vals**2.0, axis=-1) + sigma_tvals**2.0) / (2 * mass_proj)
    xvals = distance
    expected_val = compute_stopping_exact(xvals, yvals)
    data = StoppingPowerData(
        stopping=slope,
        stopping_err=slope_err,
        kinetic=yvals,
        kinetic_err=errs,
        num_samples=num_samples,
        intercept=incpt,
        intercept_err=np.sqrt(pcov[0, 0]),
        time_vals=time_vals,
        distance=distance,
        sigma_vals=np.zeros_like(kproj_vals),
        stopping_expected=expected_val,
    )
    return data


def compute_stopping_exact(xvals, yvals):
    popt, pcov = scipy.optimize.curve_fit(
        _fit_linear,
        xvals,
        yvals,
    )
    return popt[0]


@dataclass
class LSTSQ:
    popt: np.ndarray
    pcov: np.ndarray
    slope: float
    intcpt: float
    ss_res: float
    ss_tot: float
    rsq: float
    res: np.ndarray
    fx: np.ndarray


def fit_linear_stats(xvals, yvals):
    popt, pcov = scipy.optimize.curve_fit(
        _fit_linear,
        xvals,
        yvals,
    )
    fx = _fit_linear(xvals, popt[0], popt[1])
    res = yvals - fx
    ss_res = np.sum(res**2.0)
    ss_tot = np.mean((yvals - np.mean(yvals)) ** 2)
    rsquared = 1 - ss_res / ss_tot
    return LSTSQ(popt, pcov, popt[0], pcov[0], ss_res, ss_tot, rsquared, res, fx)
