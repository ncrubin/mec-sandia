import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = "arial"
plt.rcParams['font.sans-serif'] = 'Arial'
colors = ["#4285f4", "#ea4335", "#fbbc04", "#34a853"]

import numpy as np

import scipy
from scipy.special import comb

from mec_sandia.product_formulas.systems.real_space_grid import RealSpaceGrid
from mec_sandia.product_formulas.ghl_norms import compute_tau_norm, compute_nu_eta_norm

def linear(x, a, c):
    return a * x + c


def fit_linear(x, y, last_n_points=None):
    if last_n_points is None:
        last_n_points = len(x)
    if last_n_points > len(x):
        return None
    # log_x = np.log(x[-last_n_points:])
    # log_y = np.log(y[-last_n_points:])
    try:
        popt, pcov = scipy.optimize.curve_fit(linear, x, y)
        return popt
    except np.linalg.LinAlgError:
        return None
    
if __name__ == "__main__":
    ppd = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 20,  25, 30]) # np.array([2, 3, 4])#  5])
    N = ppd**3
    L = 1

    dominic_tau_values = []
    ncr_kspace_tau_values = []
    ncr_tau_values = []
    for idx, N_id in enumerate(N):
        rsg = RealSpaceGrid(L, ppd[idx])
        tau_norm = compute_tau_norm(rsg)
        tau_norm_ncr = np.max(rsg.get_kspace_h1())
        dominic_tau = 3 * ((4 * np.pi**2) / (2 * rsg.L**2)) * ((ppd[idx] - 1) / 2)**2

        print(f"{N_id:>4} & ", "\t", "{: 3.8f} & {: 3.8f} & {: 3.8f}".format(dominic_tau, tau_norm, tau_norm_ncr))
        dominic_tau_values.append(dominic_tau)
        ncr_tau_values.append(tau_norm)
        ncr_kspace_tau_values.append(tau_norm_ncr)

    params = fit_linear(np.log(ppd[-5:]), np.log(dominic_tau_values[-5:]))
    x_min = ppd[0]
    x_max = ppd[-1]
    x_vals = np.linspace(x_min, x_max, 200)
    y_vals = np.exp(params[1]) * x_vals**params[0]

    params2 = fit_linear(np.log(ppd[-5:]), np.log(ncr_tau_values[-5:]))
    y_vals2 = np.exp(params2[1]) * x_vals**params2[0]

    params3 = fit_linear(np.log(ppd[-5:]), np.log(ncr_kspace_tau_values[-5:]))
    y_vals3 = np.exp(params3[1]) * x_vals**params3[0]


    fig, ax = plt.subplots(nrows=1, ncols=1)

    ax.tick_params(which='both', labelsize=14, direction='in')

    ax.loglog(ppd, dominic_tau_values, color=colors[0], marker='o', mfc='None', markersize=8, linestyle='None', label="Analytical $\mathcal{{O}}(N^{{{:2.4f}}})$".format(params[0]))
    ax.loglog(x_vals, y_vals, color=colors[0], marker='None', linestyle='--')

    ax.loglog(ppd, ncr_tau_values, color=colors[1], marker='o', mfc='None', markersize=8, linestyle='None', label="Numerical-rspace $\mathcal{{O}}(N^{{{:2.4f}}})$".format(params2[0]))
    ax.loglog(x_vals, y_vals2, color=colors[1], marker='None', linestyle='--')

    ax.loglog(ppd, ncr_kspace_tau_values, color=colors[3], marker='x', mfc='None', markersize=8, linestyle='None', label="Numerical-kspace $\mathcal{{O}}(N^{{{:2.4f}}})$".format(params2[0]))
    ax.loglog(x_vals, y_vals3, color=colors[3], marker='None', linestyle='--')


    ax.set_xlabel(r"$N^{1/3}/\Omega^{1/3}$", fontsize=14)
    ax.set_ylabel(r"$|||\tau|||_{1}$", fontsize=14)

    ax.legend(loc='upper left', fontsize=12, ncol=1, frameon=False)
    plt.gcf().subplots_adjust(bottom=0.15, left=0.2)
    plt.savefig("tau_norm_scaling.pdf", dpi=300, format='PDF')
    plt.savefig("tau_norm_scaling.png", dpi=300, format='PNG')