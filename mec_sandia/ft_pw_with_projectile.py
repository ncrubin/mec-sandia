import numpy
from math import factorial
from sympy import factorint
from mec_sandia.ft_pw_resource_estimates import pv, eps_mt, M1, g1, h1, g2, h2, h, Eq, Er
from mec_sandia.pvec_epsmat import pvec as pv
from mec_sandia.pvec_epsmat import epsmat as eps_mt


# Probability of success for creating the superposition over 3 basis states
Peq0 = Eq(3, 8)

def pw_qubitization_with_projectile_costs(np, eta, Omega, eps, nMc, nbr, L, zeta, phase_estimation_costs=False):
    """
    :params:
       np is the number of bits in each direction for the momenta
       eta is the number of electrons
       Omega cell volume in Bohr^3
       eps is the total allowable error
       eps_mt is the precomputed discretisation errors for the nu preparation
       nMc is an adjustment for the number of bits for M (used in nu preparation
       nbr is an adjustment in the number of bits used for the nuclear positions
       L is the number of nuclei
       zeta is the charge of the projectile
       phase_estimation_costs optional (bool) return phase estimation Toffoli count and qubit costs
                              if false returns block encoding Toffoli, lambda, and num_logical qubits
    """
    # Total nuclear charge assumed to be equal to number of electrons. 
    lam_zeta = eta  
   
    # (*This is the number of bits used in rotations in preparations of equal superposition states.
    br = 7 

    # Probability of success for creating the superposition over i and j.
    # The extra + \[Zeta] is to account for the preparation with the extra 
    # nucleus treated quantum mechanically.
    Peq1 = Eq(eta + zeta, br)**2
   
    # (*Probability of success for creating the equal superposition 
    # for the selection between U and V.*)
    Peq3 = Peq0; 

    # This uses pvec from planedata.nb, which is precomputed values for
    #  \[Lambda]_\[Nu]. We start with a very large  guess for the number 
    # of bits to use for M (precision in \[Nu] \ preparation) then adjust it.*)
    p = pv[np-1,49] 
 
     # (*Now compute the lambda-values.*)
    # (*Here 64*(2^np-1))*p is \[Lambda]_\[Nu].*)
    tmp = (64*(2**np - 1)) * p / (2 * numpy.pi * Omega**(1/3))

    # (*See Eq. (D31) or (25).*)
    # tmp*(2 (\[Eta] + \[Zeta]) \[Lambda]\[Zeta] + \[Eta] (\[Eta] - 1 + 2 \[Zeta]))
    # For the case where there is the extra nucleus, the \[Lambda]_U has 
    # \[Eta] replced with \[Eta] + \[Zeta]. For \[Lambda]_V the \[Eta] (\[Eta] - 1) 
    # is replaced with (\[Eta] + \[Zeta])^2 - \[Eta] - \[Zeta]^2 = \[Eta] (\[Eta] - 1 + 2 \[Zeta]).
    # The total gives 2 (\[Eta] + \[Zeta]) \[Lambda]\[Zeta] + \[Eta] 
    # (-1 + 2 \[Zeta] + \[Eta]) used here, and the \[Eta] does not factor 
    # out so is not given in tmp as before
    lam_UV = tmp * (2 * (eta + zeta) * lam_zeta + eta * (eta - 1 + 2 * zeta))

    # (*See Eq. (25), possibly should be replaced with expression from Eq. (71).*)
    #  Here the \[Eta] is replaced with \[Eta] + \[Zeta], because we are accounting 
    # for the extra nucleus quantum mechanically. The + \[Zeta] rather than +  1 is 
    # because we are using the preparation over i, j in common with the block 
    # encoding of the potential, and there the charge of the nucleus is needed.
    lam_T = 6 * (eta + zeta) * numpy.pi**2 / Omega**(2/3) * (2**(np - 1))**2

    # (*Adjust value of nM based on \[Lambda]UV we just calculated.*) 
    nM = nMc + int(numpy.rint(numpy.log2(20 * lam_UV / eps)))

    #  (*Recompute p and \[Lambda]V.*)
    p = pv[np-1, nM-1] 
    lam_V = tmp * eta * (eta - 1 + 2 * zeta)
    lam_U = tmp * 2 * (eta + zeta) * lam_zeta

    # (*See Eq. (117).*)
    pamp = numpy.sin(3*numpy.arcsin(numpy.sqrt(p)))  

    # (*We estimate the error due to the finite M using the precomputed table.*)
    # For the extra nucleus we again replace \[Eta] (\[Eta] - 1) with \[Eta] (\[Eta] - 1 + 2 \[Zeta])
    epsM = eps_mt[np-1, nM-1] * eta * (eta - 1 + 2 * zeta) / (2 * numpy.pi * Omega**(1/3))

    # First we estimate the error due to the finite precision of the \
    # nuclear positions. 
    #   The following formula is from the formula for the error due to the \
    # nuclear positions in Theorem 4, 
    # where we have used (64*(2^np - 1))*
    #   p for the sum over 1/ | \[Nu] | .  
    #    First we estimate the number of bits to obtain an error that is \
    # some small fraction of the total error, then use that to compute the \
    # actual bound in the error for that number of bits
    nrf = (64*(2**np - 1)) * p * eta * lam_zeta / Omega**(1/3)
    nR = nbr + numpy.rint(numpy.log2(nrf/eps))

    #  (*See Eq. (133).*)
    epsR =  nrf/2**nR  

    # The number of iterations of the phase measurement. 
    # In the following the 1/(1 - 1/\[Eta]) is replaced according to the following reasoning. 
    # Note that in the discussion below Eq. (119) this expression comes from 
    # \[Eta]^2/(\[Eta] (\[Eta] - 1)) for comparing the cases with and without inequaility tests. 
    # Here we need the ratio (\[Eta] + \[Zeta])^2/(\[Eta] (\[Eta] - 1 + 2 \[Zeta])) instead
    if eps > epsM + epsR:
        eps_ph = numpy.sqrt(eps**2 - (epsM + epsR)**2)
    else:
        eps_ph = 10**(-100)

    # # (*See Eq. (127).*) 
    lam_1 = max(lam_T + lam_U + lam_V, (lam_U + lam_V * (eta + zeta)**2 / (eta * (eta - 1 + 2 * zeta))) / p) / (Peq0*Peq1* Peq3) # (*See Eq. (127).*)
    lam_2 = max(lam_T + lam_U + lam_V, (lam_U + lam_V * (eta + zeta)**2 / (eta * (eta - 1 + 2 * zeta))) / pamp) / (Peq0*Peq1*Peq3) #  (*See Eq. (126).*)
    # (*The P_eq is from Eq. (130), with P_s(\[Eta]+2\[Lambda]\[Zeta]) replaced with P_s(3,8). This is because we are taking \[Eta]=\[Lambda]\[Zeta].*)

    #  (*Steps for phase estimation without amplitude amplification.*)
    m1 = numpy.ceil(numpy.pi * lam_1 / (2 * eps_ph)) 
    m2 = numpy.ceil(numpy.pi * lam_2 / (2 * eps_ph)) 

    # (*Steps for phase estimation with amplitude amplification.*)

    # (*The number of bits used for the equal state preparation for \
    # choosing between U and V. This is significantly changed when we \
    # include the extra nucleus, because we have the relative weight 2(\
    # \[Eta]+\[Zeta])\[Lambda]\[Zeta] for \[Lambda]_U and \[Eta](\[Eta]-1+2\
    # \[Zeta]) for \[Lambda]_V, without \[Eta] factoring out. We need to \
    # prepare an equal superposition over \
    # 2(\[Eta]+\[Zeta])\[Lambda]\[Zeta]+\[Eta](\[Eta]-1+2\[Zeta]) numbers \
    # because of this.*)
    n_eta_zeta = numpy.ceil(numpy.log2(2 * (eta + zeta) * lam_zeta + eta * (eta - 1 + 2 * zeta)))
    n_eta = numpy.ceil(numpy.log2(eta + zeta))

    # We instead compute the complexity according to the complexity of 
    # preparing an equal superposition for 3 basis states, plus the 
    # complexity of rotating a qubit for T
    c1 = 2 * (n_eta_zeta + 13)

    # Here the + \[Zeta] accounts for the equal superposition including the extra nucleus
    factors = factorint(eta + zeta)
    bts = factors[min(list(sorted(factors.keys())))]

    if (eta + zeta) % 2 > 0:
        bts = 0

    # (*This is cost of superposition over i and j. See Eq. (62), or table line 2.*)
    c2 = 14 * n_eta + 8 * br - 36 - 12 * bts

    # (*Table line 3.*)
    c3 = 2 * (2 * np + 9)

    # (*Table line 4.*)
    c4 = 12 * (eta + 1) * np + 4 * (eta + 1) - 8  

    # (*Table line 5.*)
    c5 = 5 * (np - 1) + 2  

    # (*Table line 6, modified?.*)
    c6 = 3 * np**2 + 13 * np + 2 * nM * (2 * np + 2)  

    # (*The QROM cost according to the number of nuclei, line 7 modified.*)
    c7 = L + Er(L)

    c8 = 24 * np
    #  (*See Eq. (97).*)

    # c9 = 3*(Piecewise[{{2*np*nR - np*(np + 1) - 1, nR > np}}, nR*(nR - 1)]) 
    c9 = 3 * (2*np*nR - np*(np + 1) - 1 if nR > np else nR*(nR - 1))

    # (*The number of qubits we are reflecting on according to equation (136).*)
    cr = n_eta_zeta + 2 * n_eta + 6*np + nM + 16

    # (*The extra costs for accounting for the extra nucleus that is treated quantum mechanically.*)
    enuc = 30 + np + 3 + 4 * n_eta

    # (*First the cost without the amplitude amplification.*)
    cq = (c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8 + c9 + cr + enuc) * m1
    # (*Next the cost with the amplitude amplification.*)
    cqaa = (c1 + c2 + c3 + c4 + c5 + 3*c6 + c7 + c8 + c9 + cr + enuc) * m2

    # (*Qubits storing the momenta. Here \[Eta] is replaced with \[Eta]+1 for the extra nucleus.*)
    q1 = 3*(eta + 1) * np 

    q2 = 2*numpy.ceil(numpy.log2(m1 if cq < cqaa else m2)) - 1
    # (*Qubits for phase estimation.*)
    # q2 = 2*numpy.ceil(numpy.log2(Piecewise[{{m1, cq < cqaa}}, m2]]] - 1; 

    # (*We are costing WITH nuclei, so the maximum precision of rotations is nR+1.*)
    q3 =  nR + 1

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
    q9 = 3*(np + 1) + np + nM + (3*np + 2) + (2*np + 1) + (3*np**2 - np - 1 + 4*nM*(np + 1)) + 1 + 2

    # (*The nuclear positions.*)
    q10 = 3*nR     
    
    # (*Preparation of w.*)
    q11 = 4     
    
    # (*Preparation of w, r and s.*)
    q12 = 2*np + 4 
    # (*Temporary qubits for updating momenta.*)
    q13 = 5*np + 1 

    # (*Overflow bits for arithmetic on momenta.*)
    q14 = 6

    # (*Arithmetic for phasing for nuclear positions.*)
    q15 = 2*(nR - 2) 
    qt = q1 + q2 + q3 + q4 + q5 + q6 + q7 + q8 + q9 + q10 + q11 + q12 + q13 + q14

    # final_cost_toffoli, final_lambda, qpe_lam = (cq, lam_1, m1) if cq * m1 < cqaa * m2 else (cqaa, lam_2, m2)

    # return final_cost_toffoli, qt, final_lambda, qpe_lam, eps_ph
    if phase_estimation_costs:
        return min(cq, cqaa), qt
    else:
        # return block encoding cost and qubit requirement without phase estimation qubits
        if cq < cqaa:
            return cq / m1, lam_1, int(qt) - int(q2)
        else:
            return cqaa / m2, lam_2, int(qt) - int(q2)


def pw_qubitization_with_projectile_costs_from_v3(np, eta, Omega, eps, nMc, nbr, L, zeta, phase_estimation_costs=False):
    """
    :params:
       np is the number of bits in each direction for the momenta
       eta is the number of electrons
       Omega cell volume in Bohr^3
       eps is the total allowable error
       eps_mt is the precomputed discretisation errors for the nu preparation
       nMc is an adjustment for the number of bits for M (used in nu preparation
       nbr is an adjustment in the number of bits used for the nuclear positions
       L is the number of nuclei
       zeta is the charge of the projectile
       phase_estimation_costs optional (bool) return phase estimation Toffoli count and qubit costs
                              if false returns block encoding Toffoli, lambda, and num_logical qubits
    """
    # Total nuclear charge assumed to be equal to number of electrons. 
    lam_zeta = eta  
   
    # (*This is the number of bits used in rotations in preparations of equal superposition states.
    br = 7 

    # Probability of success for creating the superposition over i and j.
    # The extra + \[Zeta] is to account for the preparation with the extra 
    # nucleus treated quantum mechanically.
    Peq1 = Eq(eta + zeta, br)**2
   
    # (*Probability of success for creating the equal superposition 
    # for the selection between U and V.*)
    Peq3 = Eq(2 * (eta + zeta) * lam_zeta + eta * (eta - 1 + 2 * zeta), br)**2

    # This uses pvec from planedata.nb, which is precomputed values for
    #  \[Lambda]_\[Nu]. We start with a very large  guess for the number 
    # of bits to use for M (precision in \[Nu] \ preparation) then adjust it.*)
    p = pv[np-1,49] 
 
     # (*Now compute the lambda-values.*)
    # (*Here 64*(2^np-1))*p is \[Lambda]_\[Nu].*)
    tmp = (64*(2**np - 1)) * p / (2 * numpy.pi * Omega**(1/3))

    # (*See Eq. (D31) or (25).*)
    # tmp*(2 (\[Eta] + \[Zeta]) \[Lambda]\[Zeta] + \[Eta] (\[Eta] - 1 + 2 \[Zeta]))
    # For the case where there is the extra nucleus, the \[Lambda]_U has 
    # \[Eta] replced with \[Eta] + \[Zeta]. For \[Lambda]_V the \[Eta] (\[Eta] - 1) 
    # is replaced with (\[Eta] + \[Zeta])^2 - \[Eta] - \[Zeta]^2 = \[Eta] (\[Eta] - 1 + 2 \[Zeta]).
    # The total gives 2 (\[Eta] + \[Zeta]) \[Lambda]\[Zeta] + \[Eta] 
    # (-1 + 2 \[Zeta] + \[Eta]) used here, and the \[Eta] does not factor 
    # out so is not given in tmp as before
    lam_UV = tmp * (2 * (eta + zeta) * lam_zeta + eta * (eta - 1 + 2 * zeta))

    # (*See Eq. (25), possibly should be replaced with expression from Eq. (71).*)
    #  Here the \[Eta] is replaced with \[Eta] + \[Zeta], because we are accounting 
    # for the extra nucleus quantum mechanically. The + \[Zeta] rather than +  1 is 
    # because we are using the preparation over i, j in common with the block 
    # encoding of the potential, and there the charge of the nucleus is needed.
    lam_T = 6 * (eta + zeta) * numpy.pi**2 / Omega**(2/3) * (2**(np - 1))**2

    # (*Adjust value of nM based on \[Lambda]UV we just calculated.*) 
    nM = nMc + int(numpy.rint(numpy.log2(20 * lam_UV / eps)))

    #  (*Recompute p and \[Lambda]V.*)
    p = pv[np-1, nM-1] 
    lam_V = tmp * eta * (eta - 1 + 2 * zeta)
    lam_U = tmp * 2 * (eta + zeta) * lam_zeta

    # (*See Eq. (117).*)
    pamp = numpy.sin(3*numpy.arcsin(numpy.sqrt(p)))  

    # (*We estimate the error due to the finite M using the precomputed table.*)
    # For the extra nucleus we again replace \[Eta] (\[Eta] - 1) with \[Eta] (\[Eta] - 1 + 2 \[Zeta])
    epsM = eps_mt[np-1, nM-1] * eta * (eta - 1 + 2 * zeta) / (2 * numpy.pi * Omega**(1/3))

    # First we estimate the error due to the finite precision of the \
    # nuclear positions. 
    #   The following formula is from the formula for the error due to the \
    # nuclear positions in Theorem 4, 
    # where we have used (64*(2^np - 1))*
    #   p for the sum over 1/ | \[Nu] | .  
    #    First we estimate the number of bits to obtain an error that is \
    # some small fraction of the total error, then use that to compute the \
    # actual bound in the error for that number of bits
    nrf = (64*(2**np - 1)) * p * eta * lam_zeta / Omega**(1/3)
    nR = nbr + numpy.rint(numpy.log2(nrf/eps))

    #  (*See Eq. (133).*)
    epsR =  nrf/2**nR  

    # The number of iterations of the phase measurement. 
    # In the following the 1/(1 - 1/\[Eta]) is replaced according to the following reasoning. 
    # Note that in the discussion below Eq. (119) this expression comes from 
    # \[Eta]^2/(\[Eta] (\[Eta] - 1)) for comparing the cases with and without inequaility tests. 
    # Here we need the ratio (\[Eta] + \[Zeta])^2/(\[Eta] (\[Eta] - 1 + 2 \[Zeta])) instead
    if eps > epsM + epsR:
        eps_ph = numpy.sqrt(eps**2 - (epsM + epsR)**2)
    else:
        eps_ph = 10**(-100)

    # # (*See Eq. (127).*) 
    lam_1 = max(lam_T + lam_U + lam_V, (lam_U + lam_V * (eta + zeta)**2 / (eta * (eta - 1 + 2 * zeta))) / p) / (Peq0*Peq1* Peq3) # (*See Eq. (127).*)
    lam_2 = max(lam_T + lam_U + lam_V, (lam_U + lam_V * (eta + zeta)**2 / (eta * (eta - 1 + 2 * zeta))) / pamp) / (Peq0*Peq1*Peq3) #  (*See Eq. (126).*)
    # (*The P_eq is from Eq. (130), with P_s(\[Eta]+2\[Lambda]\[Zeta]) replaced with P_s(3,8). This is because we are taking \[Eta]=\[Lambda]\[Zeta].*)

    #  (*Steps for phase estimation without amplitude amplification.*)
    m1 = numpy.ceil(numpy.pi * lam_1 / (2 * eps_ph)) 
    m2 = numpy.ceil(numpy.pi * lam_2 / (2 * eps_ph)) 

    # (*Steps for phase estimation with amplitude amplification.*)

    # (*The number of bits used for the equal state preparation for \
    # choosing between U and V. This is significantly changed when we \
    # include the extra nucleus, because we have the relative weight 2(\
    # \[Eta]+\[Zeta])\[Lambda]\[Zeta] for \[Lambda]_U and \[Eta](\[Eta]-1+2\
    # \[Zeta]) for \[Lambda]_V, without \[Eta] factoring out. We need to \
    # prepare an equal superposition over \
    # 2(\[Eta]+\[Zeta])\[Lambda]\[Zeta]+\[Eta](\[Eta]-1+2\[Zeta]) numbers \
    # because of this.*)
    n_eta_zeta = numpy.ceil(numpy.log2(2 * (eta + zeta) * lam_zeta + eta * (eta - 1 + 2 * zeta)))
    n_eta = numpy.ceil(numpy.log2(eta + zeta))

    # We instead compute the complexity according to the complexity of 
    # preparing an equal superposition for 3 basis states, plus the 
    # complexity of rotating a qubit for T
    c1 = 2 * (n_eta_zeta + 13)

    # Here the + \[Zeta] accounts for the equal superposition including the extra nucleus
    factors = factorint(eta + zeta)
    bts = factors[min(list(sorted(factors.keys())))]

    if (eta + zeta) % 2 > 0:
        bts = 0

    # (*This is cost of superposition over i and j. See Eq. (62), or table line 2.*)
    c2 = 14 * n_eta + 8 * br - 36 - 12 * bts

    # (*Table line 3.*)
    c3 = 2 * (2 * np + 9)

    # (*Table line 4.*)
    c4 = 12 * (eta + 1) * np + 4 * (eta + 1) - 8  

    # (*Table line 5.*)
    c5 = 5 * (np - 1) + 2  

    # (*Table line 6, modified?.*)
    c6 = 3 * np**2 + 15 * np - 7 + 2 * nM * (2 * np + 2)  

    # (*The QROM cost according to the number of nuclei, line 7 modified.*)
    c7 = L + Er(L)

    c8 = 24 * np
    #  (*See Eq. (97).*)

    # c9 = 3*(Piecewise[{{2*np*nR - np*(np + 1) - 1, nR > np}}, nR*(nR - 1)]) 
    c9 = 3 * (2*np*nR - np*(np + 1) - 1 if nR > np else nR*(nR - 1))

    # (*The number of qubits we are reflecting on according to equation (136).*)
    cr = n_eta_zeta + 2 * n_eta + 6*np + nM + 16

    # (*The extra costs for accounting for the extra nucleus that is treated quantum mechanically.*)
    cnuc = 30 + np + 3 + 4 * n_eta

    # (*First the cost without the amplitude amplification.*)
    cq = (c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8 + c9 + cr + cnuc) * m1
    # (*Next the cost with the amplitude amplification.*)
    cqaa = (c1 + c2 + c3 + c4 + c5 + 3*c6 + c7 + c8 + c9 + cr + cnuc) * m2

    # (*Qubits storing the momenta. Here \[Eta] is replaced with \[Eta]+1 for the extra nucleus.*)
    q1 = 3*(eta + 1) * np 

    q2 = 2*numpy.ceil(numpy.log2(m1 if cq < cqaa else m2)) - 1
    # (*Qubits for phase estimation.*)
    # q2 = 2*numpy.ceil(numpy.log2(Piecewise[{{m1, cq < cqaa}}, m2]]] - 1; 

    # (*We are costing WITH nuclei, so the maximum precision of rotations is nR+1.*)
    q3 =  nR + 1

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
    q9 = 3*(np + 1) + np + nM + (3*np + 2) + (2*np + 1) + (3*np**2 - np - 1 + 4*nM*(np + 1)) + 1 + 2

    # (*The nuclear positions.*)
    q10 = 3*nR     
    
    # (*Preparation of w.*)
    q11 = 4     
    
    # (*Preparation of w, r and s.*)
    q12 = 2*np + 4 
    # (*Temporary qubits for updating momenta.*)
    q13 = 5*np + 1 

    # (*Overflow bits for arithmetic on momenta.*)
    q14 = 6

    # (*Arithmetic for phasing for nuclear positions.*)
    q15 = 2*(nR - 2) 
    qt = q1 + q2 + q3 + q4 + q5 + q6 + q7 + q8 + q9 + q10 + q11 + q12 + q13 + q14

    # final_cost_toffoli, final_lambda, qpe_lam = (cq, lam_1, m1) if cq * m1 < cqaa * m2 else (cqaa, lam_2, m2)

    # return final_cost_toffoli, qt, final_lambda, qpe_lam, eps_ph
    if phase_estimation_costs:
        return min(cq, cqaa), qt
    else:
        # return block encoding cost and qubit requirement without phase estimation qubits
        if cq < cqaa:
            return cq / m1, lam_1, int(qt) - int(q2)
        else:
            return cqaa / m2, lam_2, int(qt) - int(q2)



if __name__ == "__main__":
    # Let's read in the Carbon example provided by Sandia
    from mec_sandia.vasp_utils import read_vasp
    from mec_sandia.config import VASP_DATA
    import os
    
    ase_cell = read_vasp(os.path.join(VASP_DATA, "H_2eV_POSCAR"))
    # Next we can get some system paramters
    volume_ang = ase_cell.get_volume()
    print("Volume = {} A^3".format(volume_ang))
    
    # To compute rs parameter we need volume in Bohr
    from ase.units import Bohr
    volume_bohr = volume_ang / Bohr**3
    # and the number of valence electrons
    num_elec = numpy.sum(ase_cell.get_atomic_numbers())
    num_nuclei = len(numpy.where(ase_cell.get_atomic_numbers() == 1)[0])
    from mec_sandia.vasp_utils import compute_wigner_seitz_radius
    # Get the Wigner-Seitz radius
    rs = compute_wigner_seitz_radius(volume_bohr, num_elec)
    print("rs = {} bohr".format(rs))
    print("eta = {} ".format(num_elec))
    
    num_bits_momenta = 8 # Number of bits in each direction for momenta
    eps_total = 1e-3 # Total allowable error
    num_bits_nu = 8 # extra bits for nu 

    """
    :params:
       np is the number of bits in each direction for the momenta
       eta is the number of electrons
       Omega cell volume in Bohr^3
       eps is the total allowable error
       nMc is an adjustment for the number of bits for M (used in nu preparation
       nbr is an adjustment in the number of bits used for the nuclear positions
       L is the number of nuclei
       zeta is the charge of the projectile
    """
    print("eta = ", num_elec)
    print("Omega = ", volume_bohr)
    print("eps = ", eps_total)
    print("nMc = ", num_bits_nu)
    print("nbr = ", 20)
    print("L = ", num_nuclei)
    print("zeta = ", 2)
    # # final_cost_toffoli, qt, final_lambda, qpe_lam, eps_ph = \
    # qpe_cost, num_logical_qubits = \
    # pw_qubitization_with_projectile_costs(np=num_bits_momenta, 
    #                                       eta=num_elec, 
    #                                       Omega=volume_bohr, 
    #                                       eps=eps_total, 
    #                                       nMc=num_bits_nu,
    #                                       nbr=20, 
    #                                       L=num_nuclei, 
    #                                       zeta=2,
    #                                       phase_estimation_costs=True)
    # print("qpe_cost = {: 1.5e}".format(qpe_cost))
    # # assert "{:1.5e}".format(qpe_cost) == '6.29052e+13'
    # print("logical Qubit Cost ", num_logical_qubits)

    qpe_cost, num_logical_qubits = \
    pw_qubitization_with_projectile_costs_from_v3(np=num_bits_momenta, 
                                          eta=num_elec, 
                                          Omega=volume_bohr, 
                                          eps=eps_total, 
                                          nMc=num_bits_nu,
                                          nbr=20, 
                                          L=num_nuclei, 
                                          zeta=2,
                                          phase_estimation_costs=True)
    print("qpe_cost = {: 1.5e}".format(qpe_cost))
    # assert "{:1.5e}".format(qpe_cost) == '6.29494e+13'
    print("logical Qubit Cost ", num_logical_qubits)
