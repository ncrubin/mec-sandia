import numpy as np
from scipy.special import comb
from mec_sandia.product_formulas.systems.real_space_grid import RealSpaceGrid
from mec_sandia.product_formulas.ghl_norms import compute_tau_norm, compute_nu_eta_norm


if __name__ == "__main__":
    ppd = np.array([2, 3, 4])#  5])
    N = ppd**3
    eta = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    for idx, N_id in enumerate(N):
        for eta_val in eta:
            if N_id == 27 and eta_val > 10:
                continue
            elif N_id == 64 and eta_val > 6:
                continue
            if np.isclose(eta_val % 2, 0):
                hilbert_space_size = comb(N_id, eta_val//2)**2
            else:
                hilbert_space_size = comb(N_id, eta_val//2) * comb(N_id, eta_val // 2 + 1)

            memory_requirements = hilbert_space_size * 16 / (1024**3)  # 1st division bytes -> kilobytes, 2nd division kilobytes -> megabytes, 3rd division megabytes -> gigabytes

            # calculate tau and nu
            rsg = RealSpaceGrid(1., ppd[idx])
            tau_norm = compute_tau_norm(rsg)
            nu_norm = compute_nu_eta_norm(rsg, eta_val)
            
            tau_norm_ncr = np.max(rsg.get_kspace_h1())
            dominic_tau = 3 * ((4 * np.pi**2) / (2 * rsg.L**2)) * ((ppd[idx] - 1) / 2)**2
            # print(f"{N_id:>4} & {eta_val:>4}", "\t", " & {: 3.3e} & {: 3.3e} & {: 3.7f} & {: 3.7} & {: 3.7f} \\\ ".format(hilbert_space_size, memory_requirements, tau_norm, dominic_tau, tau_norm_ncr))

            dominic_nu = np.pi**(1/3) * (3./4)**(2/3) * (eta_val**(2/3) * ppd[idx] / rsg.L)
            print(f"{N_id:>4} & {eta_val:>4}", "\t", "{: 3.8f} & {: 3.8f}".format(dominic_nu, nu_norm))
