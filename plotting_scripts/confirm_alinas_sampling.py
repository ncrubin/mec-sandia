"""
This workbook sets up a simulation of the sampling bounds presented by Alina

simple line

y = sx + t

where I'm given N (y_{i}, x_{i}) pairs

Each y_{i} = <y_{i}> + epsilon where epsilon is Guassian distributed
G(o, sigma) = epsilon random variable.


"""
import numpy as np
import scipy
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
colors = ['#4285F4', '#EA4335', '#FBBC04', '#34A853']



def linear(x, a, c):
    return a * x + c


def fit_linear(x, y):
    try:
        popt, pcov = scipy.optimize.curve_fit(linear, x, y)
        return popt
    except np.linalg.LinAlgError:
        return None

def main():
    s = 3.5 
    t = 1
    n = 20
    dx = 2
    x1 = 3
    sigmasquared = 10

    x_i = lambda i: (x1 + (i - 1) * dx)
    x = np.array([x_i(ii) for ii in range(1, n+1)])
    y = s * x + t

    fig, ax = plt.subplots(nrows=1, ncols=1)

    num_trials = 100

    # for idx, sigmasquared in enumerate([5, 10, 15, 20]):
    for idx, sigmasquared in enumerate([5, 100, 1000, 10_000]):
        average_slopes = []
        average_slopes_error = []
        n_vals = []
        for n in range(1, 8): # range(2, 100):
            nn = 2**n
            n_vals.append(int(nn))
            slopes = []
            intercepts = []
            for _ in range(num_trials):
                x_i = lambda i: (x1 + (i - 1) * dx)
                x = np.array([x_i(ii) for ii in range(1, int(nn)+1)])
                y = s * x + t
                y_noise = y + np.random.normal(0, scale=np.sqrt(sigmasquared), size=int(nn))
                popt = fit_linear(x, y_noise)
                slopes.append(popt[0])
                intercepts.append(popt[1])

            average_slopes.append(np.mean(slopes))
            average_slopes_error.append(np.std(slopes, ddof=1))
            print(slopes[-1], average_slopes_error[-1])

        popt = fit_linear(np.log(n_vals), np.log(average_slopes_error))
        slopescaling = np.around(popt[0], 5)
        ax.loglog(n_vals, average_slopes_error, 'o-', label=fr'$\mathcal{{O}}(\sigma^{2}/N^{{3/2}}) = \mathcal{{O}}({{{sigmasquared}}}/N^{{{slopescaling}}})$',
                   color=colors[idx])

    ax.tick_params(which='both', labelsize=14, direction='in')
    ax.set_xlabel("$N_{t}$", fontsize=14)
    ax.set_ylabel(r"Stopping Power Error [Ha/Bohr]", fontsize=14)
    ax.tick_params(which='both', labelsize=14, direction='in')
    ax.legend(loc='upper right', fontsize=10, ncol=1, frameon=False)
    plt.gcf().subplots_adjust(bottom=0.15, left=0.2)
    plt.savefig("loglog_average_error.png", format="PNG", dpi=300)
    plt.savefig("loglog_average_error.pdf", format="PDF", dpi=300)

if __name__ == "__main__":
    main()
