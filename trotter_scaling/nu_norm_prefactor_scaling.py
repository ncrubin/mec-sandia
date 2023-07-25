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
    tau_values = []
    rsg = RealSpaceGrid(L, ppd[0])
    N_id = N[0]
    tau_norm = compute_tau_norm(rsg)
    print(tau_norm)
    dominic_tau = 3 * ((4 * np.pi**2) / (2 * rsg.L**2)) * ((ppd[0] - 1) / 2)**2
    tau_norm_ncr = np.max(rsg.get_kspace_h1())
    prefactor_values = []
    p = 2
    t = 0.65
    for eta_val in eta:
        nu_norm = compute_nu_eta_norm(rsg, eta_val)
        dominic_nu = np.pi**(1/3) * (3./4)**(2/3) * (eta_val**(2/3) * ppd[0] / rsg.L)

        # print(f"{N_id:>4} & {eta_val:>4}", "\t", "{: 3.8f} & {: 3.8f}".format(dominic_nu, nu_norm))

        dominic_nu_values.append(dominic_nu)
        ncr_nu_values.append(nu_norm)

        prefactor = (tau_norm + nu_norm) * tau_norm * nu_norm * eta_val * t**(p + 1)
        prefactor = tau_norm * tau_norm * nu_norm * eta_val * t**(p + 1) # + nu_norm * tau_norm * nu_norm * eta_val * t**(p + 1)
        prefactor = nu_norm * eta_val
        prefactor = nu_norm * nu_norm * eta_val 

        prefactor = nu_norm * eta_val + nu_norm * nu_norm * eta_val
        prefactor = (tau_norm + nu_norm) * nu_norm * eta_val
        # 59 * nu_norm * eta + nu_norm
        print(tau_norm * nu_norm * eta_val, nu_norm**2 * eta_val)

        # prefactor = nu_norm * nu_norm * eta_val

        prefactor_values.append(prefactor)

    params = fit_linear(np.log(eta), np.log(prefactor_values))
    x_min = eta[0]
    x_max = eta[-1]
    x_vals = np.linspace(x_min, x_max, 200)
    y_vals = np.exp(params[1]) * x_vals**params[0]

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.tick_params(which='both', labelsize=14, direction='in')
    ax.loglog(eta, prefactor_values, color=colors[0], marker='o', mfc='None', markersize=8, linestyle='None', label="$\mathcal{{O}}(\eta^{{{:2.4f}}})$".format(params[0]))
    ax.loglog(x_vals, y_vals, color=colors[0], marker='None', linestyle='--')

    # ax.set_title("$\Omega^{{1/3}} = {{{}}} \;,\; N^{{1/3}} = {{{}}}$".format(L, ppd[0]))

    ax.set_xlabel(r"$\eta$", fontsize=14)
    ax.set_ylabel(r"$(|||\tau|||_{1} + |||\nu|||_{1,\left[ \eta \right]})^{p-1} |||\nu|||_{1,\left[ \eta \right]} |||\tau|||_{1} \eta t^{p+1}$", fontsize=14)

    ax.legend(loc='upper left', fontsize=12, ncol=1, frameon=False)
    plt.gcf().subplots_adjust(bottom=0.15, left=0.2)
    plt.savefig("nu_norm_prefactor_scaling_smaller_system.pdf", dpi=300, format='PDF')
    plt.savefig("nu_norm_prefactor_scaling_smaller_system.png", dpi=300, format='PNG')