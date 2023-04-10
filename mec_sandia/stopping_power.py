from ase.units import Bohr, Hartree
from dataclasses import dataclass
import json
import numpy as np
import scipy.optimize

from mec_sandia.gaussians import estimate_kinetic_energy_sampling


def compute_sigma_time(time, sigma, stopping_deriv, mass_proj):
    return sigma * (1 + abs(stopping_deriv) * time / mass_proj)


def _fit_linear(x, a, b):
    return a * x + b

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
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


def parse_stopping_data(
    filename, v_proj, mass_proj=1836, num_points=20
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
    # TODO: Replace this with Alina's sampling!
    sub_sample = np.random.choice(np.arange(len(time_au)), num_points)
    time_vals = time_au[sub_sample]
    ix = np.argsort(time_vals)
    time_vals = time_vals[ix]
    kproj_x_vals = kproj_time[sub_sample][ix]
    data = DFTStoppingPowerData(
        time_au,
        position_au,
        work_au,
        ke_time,
        velocity_time,
        kproj_time,
        kproj_x_vals,
        time_vals,
        time_au*velocity_time,
    )
    return data


def compute_stopping_power(
    ecut_hartree: float,
    box_length: float,
    sigma0: int,
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
        stopping_expected=expected_val
    )
    return data

def compute_stopping_exact(xvals, yvals):
    popt, pcov = scipy.optimize.curve_fit(
        _fit_linear,
        xvals,
        yvals,
    )
    return popt[0]