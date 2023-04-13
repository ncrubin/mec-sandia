"""
Translation of slow plandata code into faster
vectorized numpy
"""
import numpy
import numexpr as ne
from tqdm import tqdm

def cartesian_prod(arrays, out=None):
    '''
    Generate a cartesian product of input arrays.
    http://stackoverflow.com/questions/1208118/using-numpy-to-build-an-array-of-all-combinations-of-two-arrays

    Args:
        arrays : list of array-like
            1-D arrays to form the cartesian product of.
        out : ndarray
            Array to place the cartesian product in.

    Returns:
        out : ndarray
            2-D array of shape (M, len(arrays)) containing cartesian products
            formed of input arrays.

    Examples:

    >>> cartesian_prod(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    '''
    arrays = [numpy.asarray(x) for x in arrays]
    dtype = numpy.result_type(*arrays)
    nd = len(arrays)
    dims = [nd] + [len(x) for x in arrays]
    out = numpy.ndarray(dims, dtype, buffer=out)

    shape = [-1] + [1] * nd
    for i, arr in enumerate(arrays):
        out[i] = arr.reshape(shape[:nd-i])

    return out.reshape(nd,-1).T


def generate_pvec_val(np, nM):
    M = 2 ** nM
    # this is equivalent to generating an ray of integers of size 2^n
    # at 2**10 this is just 8 kilobytes, at 2**20 that's just 8 MB
    axis = numpy.arange(2**np, dtype=int)
    xyz_vals = cartesian_prod([axis, axis, axis])
    sorted_xyz_vals = numpy.sort(xyz_vals, axis=-1)

    mu_vals = numpy.zeros_like(sorted_xyz_vals[:, 2], dtype=float)
    numpy.log2(sorted_xyz_vals[:, 2],
               out=mu_vals, where=sorted_xyz_vals[:, 2]>0)
    mu_vals = numpy.floor(mu_vals) + 2
    m_mu_coeff = ne.evaluate('M * 4**(mu_vals - 2)')

    # True values of prefactors
    # true_vals_prefactors = numpy.apply_along_axis(prefactor_calculator, 1, sorted_xyz_vals)
    mask = sorted_xyz_vals[:, 2] > 0
    prefactors = (mask * sorted_xyz_vals[:, 0] > 0) * 8 
    prefactors += (mask * (sorted_xyz_vals[:, 0] == 0) * (sorted_xyz_vals[:, 1] == 0 )) * 2
    prefactors += (mask * (sorted_xyz_vals[:, 0] == 0) * (sorted_xyz_vals[:, 1] > 0 )) * 4
    # assert numpy.allclose(prefactors, true_vals_prefactors)
    
    # Ceiling[M*4^(\[Mu] - 2)/(x^2 + y^2 + z^2)]
    reciprocal_values = numpy.zeros(sorted_xyz_vals.shape[0])
    numpy.reciprocal(numpy.sum(sorted_xyz_vals**2, axis=-1),
                     out=reciprocal_values,
                     where=sorted_xyz_vals[:,2]>0,
                     dtype=float
                     )
    total_vals = numpy.ceil(reciprocal_values * m_mu_coeff)

    # / M*4^(\[Mu] - 2)
    total_vals /= m_mu_coeff
    total_value = numpy.nansum(total_vals * prefactors)
    return (np, nM, total_value / (64 * (2 ** np - 1)))

def generate_epsmat_val(np, nM):
    M = 2 ** nM
    alpha =1 - 3/ (2 * M)
    # this is equivalent to generating an ray of integers of size 2^n
    # at 2**10 this is just 8 kilobytes, at 2**20 that's just 8 MB
    axis = numpy.arange(2**np, dtype=int)
    xyz_vals = cartesian_prod([axis, axis, axis])
    sorted_xyz_vals = numpy.sort(xyz_vals, axis=-1)

    mu_vals = numpy.zeros_like(sorted_xyz_vals[:, 2], dtype=float)
    numpy.log2(sorted_xyz_vals[:, 2],
               out=mu_vals, where=sorted_xyz_vals[:, 2]>0)
    mu_vals = numpy.floor(mu_vals) + 2
    m_mu_coeff = ne.evaluate('M * 4**(mu_vals - 2)')

    # True values of prefactors
    # true_vals_prefactors = numpy.apply_along_axis(prefactor_calculator, 1, sorted_xyz_vals)
    mask = sorted_xyz_vals[:, 2] > 0
    prefactors = (mask * sorted_xyz_vals[:, 0] > 0) * 8 
    prefactors += (mask * (sorted_xyz_vals[:, 0] == 0) * (sorted_xyz_vals[:, 1] == 0 )) * 2
    prefactors += (mask * (sorted_xyz_vals[:, 0] == 0) * (sorted_xyz_vals[:, 1] > 0 )) * 4
    # assert numpy.allclose(prefactors, true_vals_prefactors)
    
    # Abs[\[Alpha]* Ceiling[M*4^(\[Mu] - 2)/(x^2 + y^2 + z^2)] 
    # / N[M*4^(\[Mu] - 2), 20] - 1 / (x^2 + y^2 + z^2)]
    reciprocal_values = numpy.zeros(sorted_xyz_vals.shape[0])
    numpy.reciprocal(numpy.sum(sorted_xyz_vals**2, axis=-1),
                     out=reciprocal_values,
                     where=sorted_xyz_vals[:,2]>0,
                     dtype=float
                     )
    total_vals = alpha * numpy.ceil(reciprocal_values * m_mu_coeff)

    # / M*4^(\[Mu] - 2)
    total_vals /= m_mu_coeff
    total_vals -= reciprocal_values
    total_value = numpy.nansum(numpy.abs(total_vals * prefactors))
    return total_vals
