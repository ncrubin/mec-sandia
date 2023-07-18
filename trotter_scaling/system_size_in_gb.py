import numpy as np
from scipy.special import comb

if __name__ == "__main__":
    ppd = np.array([2, 3, 4, 5, 6, 7])
    N = ppd**3
    eta = [2, 3, 4, 5]
    for N_id in N:
        for eta_val in eta:
            if np.isclose(eta_val % 2, 0):
                hilbert_space_size = comb(N_id, eta_val//2)**2
            else:
                hilbert_space_size = comb(N_id, eta_val//2) * comb(N_id, eta_val // 2 + 1)

            memory_requirements = hilbert_space_size * 16 / (1024**3)  # 1st division bytes -> kilobytes, 2nd division kilobytes -> megabytes, 3rd division megabytes -> gigabytes
            print("{} & {} & {: 3.3e} & {: 3.3e} \\\ ".format(N_id, eta_val, hilbert_space_size, memory_requirements))
