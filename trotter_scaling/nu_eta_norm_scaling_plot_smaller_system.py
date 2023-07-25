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
    np.set_printoptions(linewidth=500)
    ppd = np.array([2]) # np.array([2, 3, 4])#  5])
    N = ppd**3
    eta = [2, 3, 4, 5, 6, 7]#  8]#  9, 10, 11, 12, 13]    
    L = 1

    dominic_nu_values = []
    ncr_nu_values = []
    for idx, N_id in enumerate(N):
        rsg = RealSpaceGrid(L, ppd[idx])
        for eta_val in eta:
            # if N_id == 27 and eta_val > 10:
            #     continue
            # elif N_id == 64 and eta_val > 6:
            #     continue
            # if np.isclose(eta_val % 2, 0):
            #     hilbert_space_size = comb(N_id, eta_val//2)**2
            # else:
            #     hilbert_space_size = comb(N_id, eta_val//2) * comb(N_id, eta_val // 2 + 1)

            # memory_requirements = hilbert_space_size * 16 / (1024**3)  # 1st division bytes -> kilobytes, 2nd division kilobytes -> megabytes, 3rd division megabytes -> gigabytes

            # calculate tau and nu
            tau_norm = compute_tau_norm(rsg)
            nu_norm = compute_nu_eta_norm(rsg, eta_val)
            tau_norm_ncr = np.max(rsg.get_kspace_h1())
            dominic_tau = 3 * ((4 * np.pi**2) / (2 * rsg.L**2)) * ((ppd[idx] - 1) / 2)**2
            # print(f"{N_id:>4} & {eta_val:>4}", "\t", " & {: 3.3e} & {: 3.3e} & {: 3.7f} & {: 3.7} & {: 3.7f} \\\ ".format(hilbert_space_size, memory_requirements, tau_norm, dominic_tau, tau_norm_ncr))

            dominic_nu = np.pi**(1/3) * (3./4)**(2/3) * (eta_val**(2/3) * ppd[idx] / rsg.L)
            print(f"{N_id:>4} & {eta_val:>4}", "\t", "{: 3.8f} & {: 3.8f}".format(dominic_nu, nu_norm))
            dominic_nu_values.append(dominic_nu)
            ncr_nu_values.append(nu_norm)

    params = fit_linear(np.log(eta), np.log(dominic_nu_values))
    x_min = eta[0]
    x_max = eta[-1]
    x_vals = np.linspace(x_min, x_max, 200)
    y_vals = np.exp(params[1]) * x_vals**params[0]

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.tick_params(which='both', labelsize=14, direction='in')
    ax.loglog(eta, dominic_nu_values, color=colors[0], marker='o', mfc='None', markersize=8, linestyle='None', label="Analytical $\mathcal{{O}}(\eta^{{{:2.4f}}})$".format(params[0]))
    ax.loglog(x_vals, y_vals, color=colors[0], marker='None', linestyle='--')


    # params2 = fit_linear(np.log(eta[-3:]), np.log(ncr_nu_values[-3:]))
    params2 = fit_linear(np.log(eta), np.log(ncr_nu_values))
    y_vals2 = np.exp(params2[1]) * x_vals**params2[0]
    ax.loglog(eta, ncr_nu_values, color=colors[1], marker='o', mfc='None', markersize=8, linestyle='None', label="Numerical $\mathcal{{O}}(\eta^{{{:2.4f}}})$".format(params2[0]))
    ax.loglog(x_vals, y_vals2, color=colors[1], marker='None', linestyle='--')

    params3 = fit_linear(np.log(eta[:3]), np.log(ncr_nu_values[:3]))
    y_vals3 = np.exp(params3[1]) * x_vals**params3[0]
    ax.loglog(x_vals, y_vals3, color='k', marker='None', alpha=0.5, linestyle='--', label="$\mathcal{{O}}(\eta^{{{:2.4f}}})$".format(params3[0]))

    ax.set_title("$\Omega^{{1/3}} = {{{}}} \;,\; N^{{1/3}} = {{{}}}$".format(L, ppd[0]))

    ax.set_xlabel(r"$\eta$", fontsize=14)
    ax.set_ylabel(r"$|||\nu|||_{1,\left[ \eta \right]}$", fontsize=14)

    ax.legend(loc='upper left', fontsize=12, ncol=1, frameon=False)
    plt.gcf().subplots_adjust(bottom=0.15, left=0.2)
    plt.savefig("nu_norm_scaling_smaller_system.pdf", dpi=300, format='PDF')
    plt.savefig("nu_norm_scaling_smaller_system.png", dpi=300, format='PNG')