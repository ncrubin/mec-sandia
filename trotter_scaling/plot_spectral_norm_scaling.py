"""
eta_2 0.41560539891787945
eta_3 0.6489071118992564
eta_4 1.2107018256097621
eta_5 1.216362270585649
eta_6 1.5696487260919076
eta_7 1.700430046965662
eta_8 1.9370379960438413
"""
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
    ppd = np.array([2]) # np.array([2, 3, 4])#  5])
    N = ppd**3
    eta = [2, 3, 4, 5, 6, 7, 8]# , 13, 14, 15]
    L = 5.
    delta_spectral_norm = np.array([0.41560539891787945,
                                    0.6489071118992564,
                                    1.2107018256097621,
                                    1.216362270585649,
                                    1.5696487260919076,
                                    1.700430046965662,
                                    1.9370379960438413])
    
    delta_spec_norms_full = np.array([[2, 0.41560539891787945],
                                      [3 , 0.648907111899256],
                                      [4 , 1.210701825609762],
                                      [5 , 1.216362270585649],
                                      [6 , 1.569648726091907],
                                      [7 , 1.700430046965662],
                                      [8 , 1.937037996043841]]) 
                                      # [9 , 1.700432575448191],
                                      # [10, 1.569647825458596],
                                      # [11, 1.216363719395915],
                                      # [12, 1.210699709012046]])
                                      # [13, 0.648860506057015],
                                      # [14, 0.415606295294775],
                                      # [15, 4.4846949524631364e-14],
                                      # [16, 3.966683503312497e-14]])
    delta_spectral_norm = delta_spec_norms_full[:, 1]

    rsg = RealSpaceGrid(L, ppd[0])
    strang_constant_factors = []
    evol_time = 0.65
    for eta_idx, eta_val in enumerate(eta):
        tau_norm = compute_tau_norm(rsg)
        nu_norm = compute_nu_eta_norm(rsg, eta_val)


        prefactor = ((tau_norm + nu_norm) * tau_norm * nu_norm * evol_time**(3))
        constant_factor = delta_spectral_norm[eta_idx] / prefactor
        strang_constant_factors.append(constant_factor)

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.tick_params(which='both', labelsize=14, direction='in')
    ax.plot(eta, strang_constant_factors, color=colors[0], marker='o', linestyle='None', label='p=2')
    ax.set_xlabel(r"$\eta$", fontsize=14)
    ax.set_ylabel(r"$||S_{p}(t) - e^{-itH}||_{\mathcal{W}_{\eta}} /$ Prefactor", fontsize=14)
    ax.legend(fontsize=12, ncol=1, frameon=False)
    plt.gcf().subplots_adjust(bottom=0.15, left=0.2)
    plt.savefig("constant_factor_grid_scaling.png", dpi=300, format='PNG')





