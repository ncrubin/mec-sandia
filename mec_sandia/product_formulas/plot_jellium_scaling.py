import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
colors = ['#4285F4', '#EA4335', '#FBBC04', '#34A853']

import scipy
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
    
def main():
    # strang_norms = np.load("strang_spectral_norms.npy")
    # suzuki_4_norms = np.load("suzuki_4_spectral_norms.npy")
    # suzuki_6_norms = np.load("suzuki_6_spectral_norms.npy")
    # berry_norms = np.load("berry_spectral_norms.npy")
    # tvals = np.logspace(0, -4, 10)
    # tvals = np.logspace(0, -3, 10)
    # tvals = np.logspace(-1, -3, 10)
    # tvals = np.logspace(-0.5, -2, 10)

    berry_norms = np.load("berry_spectral_norms.npy")
    cirq_norms = np.load("cirq_spectral_norms.npy")
    tvals = np.load("tvals.npy")


    # params = fit_linear(np.log(tvals), np.log(berry_norms))
    x_vals = np.logspace(np.log10(tvals)[0], np.log10(tvals)[-1], 50)
    # y_vals = np.exp(params[1]) * x_vals**params[0]

    params2 = fit_linear(np.log(tvals), np.log(cirq_norms))
    x_vals2 = np.logspace(np.log10(tvals)[0], np.log10(tvals)[-1], 50)
    y_vals2 = np.exp(params2[1]) * x_vals**params2[0]



    fig, ax = plt.subplots(nrows=1, ncols=1)
    # ax.loglog(tvals, berry_norms, color=colors[3], mfc='None', mec=colors[3], marker='o', linestyle='-', label='Berry-8')
    # ax.loglog(x_vals, y_vals, color=colors[3], linestyle='--', label=r"$\mathcal{{O}}(t^{{{:2.2f}}})$".format(params[0]))
    ax.loglog(tvals, cirq_norms, color=colors[2], mfc='None', mec=colors[2], marker='o', linestyle='-', label='Cirq-Berry-8')
    ax.loglog(x_vals2, y_vals2, color=colors[2], linestyle='--', label=r"$\mathcal{{O}}(t^{{{:2.2f}}})$".format(params2[0]))

 
    ax.tick_params(which='both', labelsize=14, direction='in')
    ax.set_xlabel("$t\;[U(t)]$", fontsize=14)
    ax.set_ylabel(r"$||U_{\mathrm{prod}}(t) - U_{\mathrm{exact}}(t)||$", fontsize=14)
    ax.tick_params(which='both', labelsize=14, direction='in')
    ax.legend(loc='upper left', fontsize=14, ncol=2, frameon=False)
    plt.gcf().subplots_adjust(bottom=0.15, left=0.2)
    plt.savefig("Jellium_4elec_7_orb.png", format='PNG', dpi=300)

if __name__ == "__main__":
    main()