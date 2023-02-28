import numpy
from math import factorial
from sympy import factorint


def M1(k, K):
    return numpy.ceil(numpy.log2(factorial(K) * numpy.sum([1/factorial(k1) for  k1 in range(k,  K+1)]
                                                )
                          )
                   )

def g1(x, n):
    asin_val = 0.5 / numpy.sqrt(x)
    floored_val = numpy.floor(2**n * asin_val / (2 * numpy.pi))
    return floored_val * 2 * numpy.pi / 2**n

def h1(x, n):
    return x * ((1 + (2 - 4 * x) * numpy.sin(g1(x, n))**2)**2 + 4*numpy.sin(g1(x, n))**2 * numpy.cos(g1(x, n))**2)

def g2(x, n):
    asin_val = numpy.arcsin(0.5 / numpy.sqrt(x)) 
    return numpy.ceil(2**n * asin_val / (2 * numpy.pi)) * 2 * numpy.pi / 2**n

def h2(x, n):
    return x * ((1 + (2 - 4 * x) * numpy.sin(g2(x, n))**2)**2 + 4 * numpy.sin(g2(x, n))**2 * numpy.cos(g2(x, n))**2)

def h(x, n):
    return numpy.max([h1(x, n), h2(x, n)])

def Eq(n, br):
    return h(n / 2**(numpy.ceil(numpy.log2(n))), br)
    
def Er(zeta):
    kt1 = 2**numpy.floor(numpy.log2(zeta)/2)
    kt2 = 2**numpy.ceil(numpy.log2(zeta)/2)
    return numpy.min([numpy.ceil(zeta / kt1) + kt1, 
                   numpy.ceil(zeta / kt2) + kt2]
                 )

# Probability of success for creating the superposition over 3 basis states
Peq0 = Eq(3, 8)

def pw_qubitization_costs(np, eta, Omega, eps, pv, eps_mt, nMc, nbr, L):
    """
    :params:
       lam_zeta is the sum over nuclear weights
       np is the number of bits in each direction for the momenta
       eta is the number of electrons
       rs is the Wigner-Seitz radius
       eps is the total allowable error
       pv is the precomputed vector of probabilities of success for the nu preparation
       eps_mt is the precomputed discretisation errors for the nu preparation
       nMc is an adjustment for the number of bits for M (used in nu preparation
       ntc is an adjustment in the number of bits used for the time
       nbr is an adjustment in the number of bits used for the nuclear positions
       L is the number of nuclei
    """
    # Total nuclear charge assumed to be equal to number of electrons. 
    lam_zeta = eta  
    
    # (*This is the number of bits used in rotations in preparations of equal superposition states.
    br = 7 
    
    # The following uses the precomputed table to find the exact value of p based on np.
    
    # (*Probability of success for creating the superposition over i and j.*)
    Peq1 = Eq(eta, br)**2
    
    # (*Probability of success for creating the equal superposition for the selection between U and V.*)
    Peq3 = Peq0; 
    
    # This uses pvec from planedata.nb, which is precomputed values for
    #  \[Lambda]_\[Nu]. We start with a very large  guess for the number 
    # of bits to use for M (precision in \[Nu] \ preparation) then adjust it.*)
    p = pv[np, 50]
    
    # (*Now compute the lambda-values.*)
    # (*Here 64*(2^np-1))*p is \[Lambda]_\[Nu].*)
    tmp = (64*(2**np - 1)) * p * eta / (2 * numpy.pi * Omega**(1/3))
    
    # (*See Eq. (D31) or (25).*)
    lam_UV = tmp * (eta - 1 + 2 * lam_zeta)
    
    # (*See Eq. (25), possibly should be replaced with expression from Eq. (71).*)
    lam_T =  6 * eta * numpy.pi**2 / Omega**(2/3) * (2**(np - 1) - 1)**2
    
    # (*Adjust value of nM based on \[Lambda]UV we just calculated.*) 
    nM = nMc + numpy.rint(numpy.log2(20 * lam_UV / eps));   
    
    #  (*Recompute p and \[Lambda]V.*)
    p = pv[np, nM] 
    lam_V = tmp * (eta - 1)
    lam_U = tmp * 2 * lam_zeta
    
    # (*See Eq. (117).*)
    pamp = numpy.sin(3*numpy.arcsin(numpy.sqrt(p)))  
    
    # (*We estimate the error due to the finite M using the precomputed table.*)
    epsM = eps_mt[np, nM] * eta * (eta - 1) / (2 * numpy.pi * Omega**(1/3))
    
    # (*First we estimate the error due to the finite precision of the \
    # nuclear positions. The following formula is from the formula for the \
    # error due to the nuclear positions in Theorem 4, where we have used \
    # (64*(2^np-1))*p for the sum over 1/|\[Nu]|.  First we estimate the \
    # number of bits to obtain an error that is some small fraction of the \
    # total error, then use that to compute the actual bound in the error \
    # for that number of bits.*)
    nrf = (64*(2**np - 1)) * p * eta * lam_zeta / Omega**(1/3)
    nR = nbr + numpy.rint(numpy.log2(nrf/eps));
    
    #  (*See Eq. (133).*)
    epsR =  nrf/2**nR  
    # (*Set the allowable error in the phase measurement such that the sum of the squares in the errors is \[Epsilon]^2, as per Eq. (131).*)
    
    if eps > epsM + epsR:
        eps_ph = numpy.sqrt(eps**2 - (epsM + epsR)**2)
    else:
        eps_ph = 10**(-100)
    # (*The number of iterations of the phase measurement.*)
    
    # # (*See Eq. (127).*) 
    lam_1 = numpy.max(lam_T + lam_U + lam_V, (lam_U + lam_V / (1 - 1 / eta)) / p) / (Peq0 * Peq1* Peq3) 
    lam_2 = numpy.max(lam_T + lam_U + lam_V, (lam_U + lam_V / (1 - 1 / eta)) /pamp) / (Peq0 * Peq1 * Peq3)
    # (*See Eq. (126).*)
    # (*The P_eq is from Eq. (130), with P_s(\[Eta]+2lam_zeta) replaced with P_s(3,8). This is because we are taking \[Eta]=lam_zeta.*)
    #  (*Steps for phase estimation without amplitude amplification.*)
    m1 = numpy.ceil(numpy.pi * lam_1 / (2 * eps_ph)) 
    m2 = numpy.ceil(numpy.pi * lam_2 / (2 * eps_ph)) 
    # (*Steps for phase estimation with amplitude amplification.*)

    # (*The number of bits used for the equal state preparation for choosing between U and V.*)
    n_eta_zeta = numpy.ceil(numpy.log2(eta + lam_zeta))
    n_eta = numpy.ceil(numpy.log2(eta));
    # (*Set the costs of the parts of the block encoding according to the list in table II.*)

    # (*c1=2*(5*n\[Eta]\[Zeta]+2*br-9);
    # We instead compute the complexity according to the complexity of \
    # preparing an equal superposition for 3 basis states, plus the \
    # complexity of rotating a qubit for T.*)
    c1 = 2 * (n_eta_zeta + 13)
    # (*c2=14*n\[Eta]+8*br-36;*)
    factors = factorint(eta)
    bts = factors[min(list(sorted(factors.keys())))]
    # bts = FactorInteger[\[Eta]][[1, 2]];
    if eta % 2 > 0:
        bts = 0

    # (*This is cost of superposition over i and j. See Eq. (62), or table line 2.*)
    c2 = 14 * n_eta + 8 * br - 36 - 12 * bts
    # (*Table line 3.*)
    c3 = 2 * (2 * np + 9)
    # (*Table line 4.*)
    c4 = 12 * eta * np + 4 * eta - 8  
    # (*Table line 5.*)
    c5 = 5 * (np - 1) + 2  
    # (*Table line 6, modified?.*)
    c6 = 3 * np**2 + 13 * np - 10 + 2 * nM * (2 * np + 2)  

    # (*The QROM cost according to the number of nuclei, line 7 modified.*)
    c7 = L + Er(L)
    # (*Line 8.*)
    c8 = 24 * np
    #  (*See Eq. (97).*)

    # c9 = 3*(Piecewise[{{2*np*nR - np*(np + 1) - 1, nR > np}}, nR*(nR - 1)]) 
    c9 = 3 * (2*np*nR - np*(np + 1) - 1 if nR > np else nR*(nR - 1))


    # (*The number of qubits we are reflecting on according to equation (136).*)
    cr = n_eta_zeta + 2 * n_eta + 6*np + nM + 16

    # (*First the cost without the amplitude amplification.*)
    cq = (c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8 + c9 + cr) * m1
    # (*Next the cost with the amplitude amplification.*)
    cqaa = (c1 + c2 + c3 + c4 + c5 + 3*c6 + c7 + c8 + c9 + cr)*m2

    # (*Qubits for qubitisation.*)
    q1 = 3 * eta * np # (*Qubits storing the momenta.*)

    # (*Qubits for phase estimation.*)
    # q2 = 2*numpy.ceil(numpy.log2(Piecewise[{{m1, cq < cqaa}}, m2]]] - 1 
    q2 = 2*numpy.ceil(numpy.log2(m1 if cq < cqaa else m2)) - 1

    # (*We are costing WITH nuclei, so the maximum precision of rotations is nR+1.*)
    q3 = nR + 1 
    # (*The |T> state.*)
    q4 = 1 
    # (*The rotated qubit for T vs U+V.*)
    q5 = 1 
    # (*The superposition state for selecting between U and V. This is changed from n\[Eta]\[Zeta]+3 to bL+4, with Log2[L] for outputting L.*)
    q6 = numpy.ceil(numpy.log2(L)) + 4 

    # (*The result of a Toffoli on the last two.*)
    q7 = 1 

    # (*Preparing the superposition over i and j.*)
    q8 = 2 * n_eta + 5 
    # (*For preparing the superposition over \[Nu].*)
    q9 = 3*(np + 1) + np + nM + (3*np + 2) + (2*np + 1) + (3*np^2 - np - 1 + 4*nM*(np + 1)) + 1 + 2

    # (*The nuclear positions.*)
    q10 = 3*nR 
    # (*Preparation of w.*)
    q11 = 4 
    # (*Preparation of w, r and s.*)
    q12 =2*np + 4 
    # (*Temporary qubits for updating momenta.*)
    q13 = 5*np + 1 
    # (*Overflow bits for arithmetic on momenta.*)
    q14 = 6
    # (*Arithmetic for phasing for nuclear positions.*)
    q15 = 2*(nR - 2) 
    qt = q1 + q2 + q3 + q4 + q5 + q6 + q7 + q8 + q9 + q10 + q11 + q12 + q13 + q14
    return numpy.min([cq, cqaa]), qt