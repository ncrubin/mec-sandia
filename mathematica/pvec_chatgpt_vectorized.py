import numpy
import time

maxbitsize = 4
pvec = numpy.zeros((maxbitsize, 50))
start_time = time.time()

n_range = numpy.arange(1, maxbitsize + 1)
M_range = 2 ** numpy.arange(1, 51)

for i, np in numpy.ndenumerate(n_range):
    for j, nM in numpy.ndenumerate(M_range):
        alpha = 1 - 3 / (2 * nM)
        tot = 0
        x_range = numpy.arange(2 ** np)
        y_range = numpy.arange(2 ** np)
        z_range = numpy.arange(2 ** np)
        xx, yy, zz = numpy.meshgrid(x_range, y_range, z_range, indexing='ij')
        s = numpy.sort(numpy.stack((xx, yy, zz), axis=-1), axis=-1)
        mu = numpy.floor(numpy.log2(s[:, :, :, 2])) + 2
        mask1 = s[:, :, :, 2] > 0
        mask2 = s[:, :, :, 0] != 0
        mask3 = s[:, :, :, 1] == 0
        mask4 = ~(mask2 | mask3)
        tot += numpy.sum(8 * numpy.abs(numpy.ceil((nM * 4 ** (mu - 2)) / (xx ** 2 + yy ** 2 + zz ** 2))) / (nM * 4 ** (mu - 2)) * mask1 * mask2)
        tot += numpy.sum(2 * numpy.abs(numpy.ceil((nM * 4 ** (mu - 2)) / (xx ** 2 + yy ** 2 + zz ** 2))) / (nM * 4 ** (mu - 2)) * mask1 * mask3)
        tot += numpy.sum(4 * numpy.abs(numpy.ceil((nM * 4 ** (mu - 2)) / (xx ** 2 + yy ** 2 + zz ** 2))) / (nM * 4 ** (mu - 2)) * mask1 * mask4)
        pvec[i, j] = tot / (64 * (2 ** np - 1))

elapsed_time = time.time() - start_time
print("Elapsed time: {:.2f} seconds".format(elapsed_time))

from mec_sandia.ft_pw_resource_estimates import pv as pvec_dominic
print(pvec_dominic[:3, :10])
print(pvec[:3, :10])
assert numpy.allclose(pvec_dominic[:3, :10], pvec[:3, :10])

