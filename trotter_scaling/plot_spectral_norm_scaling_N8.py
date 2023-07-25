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
    strang_delta_spectral_norm = np.array([0.41560539891787945,
                                    0.6489071118992564,
                                    1.2107018256097621,
                                    1.216362270585649,
                                    1.5696487260919076,
                                    1.700430046965662,
                                    1.9370379960438413])
    
    strang_delta_spec_norms_full = np.array([[2, 0.41560539891787945],
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
    strang_delta_spectral_norm = strang_delta_spec_norms_full[:, 1]

    suzuki_4_delta_spectral_norm = np.array([0.356741302140853,
                                           0.460613370108645,
                                           0.841408096435836,
                                           0.865653733326057,
                                           1.138355451859550,
                                           1.288682205360836,
                                           1.556014865877181,
    ])

    suzuki_6_delta_spectral_norm = np.array([0.1311016304979491,
                                             0.1357750271167295,
                                             0.1470436920921759,
                                             0.1487035457272558,
                                             0.1657424087380123,
                                             0.1764095639271601,
                                             0.1952451085199114,
                                             ])

    berry_8_delta_spectral_norm = np.array([0.0646806948143044,
                                            0.0939773551204049,
                                            0.1593559313465294,
                                            0.1599381658229178,
                                            0.1873571861962500,
                                            0.1928124097178196,
                                            0.22085871665304704])

    rsg = RealSpaceGrid(L, ppd[0])
    strang_constant_factors = []
    strang_prefactors = []
    suzuki_4_constant_factors = []
    suzuki_4_prefactors = []
    suzuki_6_constant_factors = []
    suzuki_6_prefactors = []
    berry_8_constant_factors = []
    berry_8_prefactors = []

    evol_time = 0.65
    tau_norm = compute_tau_norm(rsg)
    for eta_idx, eta_val in enumerate(eta):
        nu_norm = compute_nu_eta_norm(rsg, eta_val)

        p = 2
        prefactor = (tau_norm + nu_norm)**(p - 1) * tau_norm * nu_norm * evol_time**(p + 1)  * eta_val
        strang_prefactors.append(prefactor)
        constant_factor = strang_delta_spectral_norm[eta_idx] / prefactor
        strang_constant_factors.append(constant_factor)

        p = 4
        prefactor = (tau_norm + nu_norm)**(p - 1) * tau_norm * nu_norm * evol_time**(p + 1) * eta_val
        suzuki_4_prefactors.append(prefactor)
        constant_factor = suzuki_4_delta_spectral_norm[eta_idx] / prefactor
        suzuki_4_constant_factors.append(constant_factor)

        p = 6
        prefactor = (tau_norm + nu_norm)**(p - 1) * tau_norm * nu_norm * evol_time**(p + 1)  * eta_val
        suzuki_6_prefactors.append(prefactor)
        constant_factor = suzuki_6_delta_spectral_norm[eta_idx] / prefactor
        suzuki_6_constant_factors.append(constant_factor)

        p = 8
        prefactor = (tau_norm + nu_norm)**(p - 1) * tau_norm * nu_norm * evol_time**(p + 1)  * eta_val
        berry_8_prefactors.append(prefactor)
        constant_factor = berry_8_delta_spectral_norm[eta_idx] / prefactor
        berry_8_constant_factors.append(constant_factor)


    # fig, ax = plt.subplots(nrows=1, ncols=1)
    # ax.tick_params(which='both', labelsize=14, direction='in')
    # # ax.plot(eta, strang_constant_factors, color=colors[0], marker='o', linestyle='None', label='$p=2, N=8$')
    # # ax.plot(eta, suzuki_4_constant_factors, color=colors[1], marker='o', linestyle='None', label='$p=4, N=8$')
    # ax.plot(eta, suzuki_6_constant_factors, color=colors[2], marker='o', linestyle='None', label='$p=6, N=8$')
    # # ax.plot(eta, berry_8_constant_factors, color=colors[3], marker='o', linestyle='None', label='$p^{*}=8, N=8$')
    # ax.set_xlabel(r"$\eta$", fontsize=14)
    # ax.set_ylabel(r"$||S_{p}(t) - e^{-itH}||_{\mathcal{W}_{\eta}} /$ Prefactor", fontsize=14)
    # ax.legend(fontsize=12, ncol=1, frameon=False)
    # plt.gcf().subplots_adjust(bottom=0.15, left=0.2)
    # plt.savefig("constant_factor_grid_scaling_N8.png", dpi=300, format='PNG')


    ###############################
    #
    # Plot spectral norms
    #
    ###############################
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
    ax[0, 0].tick_params(which='both', labelsize=14, direction='in')
    ax[0, 1].tick_params(which='both', labelsize=14, direction='in')
    ax[1, 0].tick_params(which='both', labelsize=14, direction='in')
    ax[1, 1].tick_params(which='both', labelsize=14, direction='in')


    x_vals = np.linspace(eta[0], eta[-1], 100)
    params_strang = fit_linear(eta[-3:], strang_delta_spectral_norm[-3:])
    y_vals = params_strang[0] * x_vals + params_strang[1]
    ax[0, 0].plot(x_vals, y_vals, color=colors[0], linestyle='--', alpha=0.5)
    ax[0, 0].plot(eta, strang_delta_spectral_norm, color=colors[0], marker='o', linestyle='None', label="$p=2, N=8$")
    # ax[0, 0].set_xlabel(r"$\eta$", fontsize=14)
    ax[0, 0].set_ylabel(r"$||S_{p}(t) - e^{-itH}||_{\mathcal{W}_{\eta}}$", fontsize=14)
    ax[0, 0].legend(fontsize=12, ncol=1, frameon=False)


    params_suzuki_4 = fit_linear(eta[-3:], suzuki_4_delta_spectral_norm[-3:])
    y_vals = params_suzuki_4[0] * x_vals + params_suzuki_4[1]
    ax[0, 1].plot(x_vals, y_vals, color=colors[1], linestyle='--', alpha=0.5)
    ax[0, 1].plot(eta, suzuki_4_delta_spectral_norm, color=colors[1], marker='o', linestyle='None', label='$p=4, N=8$')
    # ax[0, 1].set_xlabel(r"$\eta$", fontsize=14)
    # ax[0, 1].set_ylabel(r"$||S_{p}(t) - e^{-itH}||_{\mathcal{W}_{\eta}}$", fontsize=14)
    ax[0, 1].legend(fontsize=12, ncol=1, frameon=False)


    params_suzuki_6 = fit_linear(eta[-3:], suzuki_6_delta_spectral_norm[-3:])
    y_vals = params_suzuki_6[0] * x_vals + params_suzuki_6[1]
    ax[1, 0].plot(x_vals, y_vals, color=colors[2], linestyle='--', alpha=0.5)
    ax[1, 0].plot(eta, suzuki_6_delta_spectral_norm, color=colors[2], marker='o', linestyle='None', label='$p=6, N=8$')
    ax[1, 0].set_ylabel(r"$||S_{p}(t) - e^{-itH}||_{\mathcal{W}_{\eta}}$", fontsize=14)
    ax[1, 0].set_xlabel(r"$\eta$", fontsize=14)
    ax[1, 0].legend(fontsize=12, ncol=1, frameon=False)


    params_berry_8 = fit_linear(eta[-3:], berry_8_delta_spectral_norm[-3:])
    y_vals = params_berry_8[0] * x_vals + params_berry_8[1]
    ax[1, 1].plot(x_vals, y_vals, color=colors[3], linestyle='--', alpha=0.5)
    ax[1, 1].plot(eta, berry_8_delta_spectral_norm, color=colors[3], marker='o', linestyle='None', label='$p^{*}=8, N=8$')
    ax[1, 1].set_xlabel(r"$\eta$", fontsize=14)
    ax[1, 1].legend(fontsize=12, ncol=1, frameon=False)
    plt.gcf().subplots_adjust(bottom=0.15, left=0.2)
    plt.savefig("delta_spectral_norms_grid_scaling_N8.png", dpi=300, format='PNG')





    ###############################
    #
    # Plot scaled spectral norms
    #
    ###############################

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
    ax[0, 0].tick_params(which='both', labelsize=14, direction='in')
    ax[0, 1].tick_params(which='both', labelsize=14, direction='in')
    ax[1, 0].tick_params(which='both', labelsize=14, direction='in')
    ax[1, 1].tick_params(which='both', labelsize=14, direction='in')


    x_vals = np.linspace(eta[0], eta[-1], 100)
    params_strang = fit_linear(eta[-3:], strang_constant_factors[-3:])
    y_vals = params_strang[0] * x_vals + params_strang[1]
    ax[0, 0].plot(x_vals, y_vals, color=colors[0], linestyle='--', alpha=0.5)
    ax[0, 0].plot(eta, strang_constant_factors, color=colors[0], marker='o', linestyle='None', label="$p=2, N=8$")
    # ax[0, 0].set_xlabel(r"$\eta$", fontsize=14)
    ax[0, 0].set_ylabel(r"$||S_{p}(t) - e^{-itH}||_{\mathcal{W}_{\eta}}$ / prefactor", fontsize=14)
    ax[0, 0].legend(fontsize=12, ncol=1, frameon=False)


    params_suzuki_4 = fit_linear(eta[-3:], suzuki_4_constant_factors[-3:])
    y_vals = params_suzuki_4[0] * x_vals + params_suzuki_4[1]
    ax[0, 1].plot(x_vals, y_vals, color=colors[1], linestyle='--', alpha=0.5)
    ax[0, 1].plot(eta, suzuki_4_constant_factors, color=colors[1], marker='o', linestyle='None', label='$p=4, N=8$')
    ax[0, 1].legend(fontsize=12, ncol=1, frameon=False)


    params_suzuki_6 = fit_linear(eta[-3:], suzuki_6_constant_factors[-3:])
    y_vals = params_suzuki_6[0] * x_vals + params_suzuki_6[1]
    ax[1, 0].plot(x_vals, y_vals, color=colors[2], linestyle='--', alpha=0.5)
    ax[1, 0].plot(eta, suzuki_6_constant_factors, color=colors[2], marker='o', linestyle='None', label='$p=6, N=8$')
    ax[1, 0].set_ylabel(r"$||S_{p}(t) - e^{-itH}||_{\mathcal{W}_{\eta}}$ / prefactor", fontsize=14)
    ax[1, 0].set_xlabel(r"$\eta$", fontsize=14)
    ax[1, 0].legend(fontsize=12, ncol=1, frameon=False)


    params_berry_8 = fit_linear(eta[-3:], berry_8_constant_factors[-3:])
    y_vals = params_berry_8[0] * x_vals + params_berry_8[1]
    ax[1, 1].plot(x_vals, y_vals, color=colors[3], linestyle='--', alpha=0.5)
    ax[1, 1].plot(eta, berry_8_constant_factors, color=colors[3], marker='o', linestyle='None', label='$p^{*}=8, N=8$')
    ax[1, 1].set_xlabel(r"$\eta$", fontsize=14)
    ax[1, 1].legend(fontsize=12, ncol=1, frameon=False)
    plt.gcf().subplots_adjust(bottom=0.15, left=0.2)
    plt.savefig("delta_scaled_spectral_norm_grid_scaling_N8.png", dpi=300, format='PNG')



