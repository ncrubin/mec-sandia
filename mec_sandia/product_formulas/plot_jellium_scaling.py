import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
colors = ['#4285F4', '#EA4335', '#FBBC04', '#34A853']

def main():
    strang_norms = np.load("strang_spectral_norms.npy")
    suzuki_4_norms = np.load("suzuki_4_spectral_norms.npy")
    suzuki_6_norms = np.load("suzuki_6_spectral_norms.npy")
    berry_norms = np.load("berry_spectral_norms.npy")
    tvals = np.logspace(0, -4, 10)
    tvals = np.logspace(0, -3, 10)
    tvals = np.logspace(-1, -3, 10)
    tvals = np.logspace(-0.5, -2, 10)

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.loglog(tvals, strang_norms, color=colors[0], mfc='None', mec=colors[0], marker='o', linestyle='-', label='Strang')
    ax.loglog(tvals, suzuki_4_norms, color=colors[1], mfc='None', mec=colors[1], marker='o', linestyle='-', label='Suzuki-4')
    ax.loglog(tvals, suzuki_6_norms, color=colors[2], mfc='None', mec=colors[2], marker='o', linestyle='-', label='Suzuki-6')
    ax.loglog(tvals, berry_norms, color=colors[3], mfc='None', mec=colors[3], marker='o', linestyle='-', label='Berry-10')
    ax.tick_params(which='both', labelsize=14, direction='in')
    ax.set_xlabel("$t\;[U(t)]$", fontsize=14)
    ax.set_ylabel(r"$||U_{\mathrm{prod}}(t) - U_{\mathrm{exact}}(t)||$", fontsize=14)
    ax.tick_params(which='both', labelsize=14, direction='in')
    ax.legend(loc='upper left', fontsize=14, ncol=1, frameon=False)
    plt.gcf().subplots_adjust(bottom=0.15, left=0.2)
    plt.savefig("Jellium_4elec_7_orb.png", format='PNG', dpi=300)

if __name__ == "__main__":
    main()