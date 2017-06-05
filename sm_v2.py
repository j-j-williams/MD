"""
This notebook provides the defenitions and descriptions for various plasma parameters
relevant to transport properties in plasma.  Most of these fourmula are derived from
the paper of L. G. Stanton and M. S. Murillo "Ionic transport in high-energy-density
matter", PRE ${\bf 93}$ 043203 (2016).  I will include the relevant equations number
and descripton from the text in the documentaion below.
"""

import numpy as np
qe    = np.sqrt(1.4399864)  # Unit of electric charge in convient units of [MeV*fm]
q2 = qe*qe*1e6*1e-13      # n is in cm^-3 and qe^2 is MeV*fm so I need to fix lengths and energy  eV*cm

#error 1 - this was 10e6*10e-13, should be 1e6 and 1e-13
#This would give q2 = 1.44e-7, the previous formula gives q2 = 1.44e-5

bohr  = 5.2918e-9           # The classical bohr radius in [cm] for the length scale in atomic untis in LAMMPS
hbar = 6.582e-16            # in eV*s
me = 511000                 # electron mass in eV units
c  = 29979245800.00         # speed of light in cm/s
mp = 1.6726e-24             # proton mass in grams
mn = 1.6749e-24             # neutron mass in grams
AMU = 1.6605e-24            # atomic mass unit
m_e = 9.1094e-28            # electron mass in grams

#Added unit for conversion from eV
erg_to_ev = 6.2415e11

def factorial (n):
    """
	Factorial function:
	Output:  n!
    """
    if (n <= 1):
        return 1

    i = 1
    product = 1
    while (i <= n):
        product = product * i
        i = i + 1
    return product
    
def lambdaDH(T, n, Z):
    """
    Start with the Debye-Hukel screening length in simplest form.  
    $\lambda_{DH} = \sqrt{T/4 \pi n Z^2 e^2}$.
    """
    denom = 4.*np.pi*n*Z*Z*q2    #eV/cm^2
    ans2  = T/denom              #cm^2
    ans   = np.sqrt(ans2)        #cm
    return ans                   # in cm
    
def ai(n):
    """
    $a_i=(3/4\pi n_i)^{1/3}$ is the ionic radius.
    """
    ans = (3./4./np.pi/n)**(1./3.) #cm
    return ans    # in cm
    
def gamma(T,n,Z):
    """
    $\Gamma = Z^2 e^2/a_i T$
    """
    num   = Z*Z*q2      #eV*cm
    a = ai(n)           #cm
    denom = a*T         #eV * cm
    ans = num/denom     #unitless
    return ans # this is dimensionsless
    
def logLambda(T,n,Z):
    """
    A standard definition for the Coulomb logorithm is given by 
    $ Ln(\Lambda) \approx ln(1/\sqrt{3} \Gamma^{3/2})$
    """
    gam = gamma(T,n,Z)
    eans = 1./(np.sqrt(3.)*gam**(1.5))
    if gam < 0.69:
        ans = np.log(eans)
    else:
        print('error in calculating log lambda as log becoming negative, \
        too strongly coupled for this definition')
        return
    return ans  # this is dimensionless
    
def EF(n):
    """
    $E_F = c^2 \hbar^2(3\pi^2 n_e)^{2/3}/2m_ec^2$
    """
    hbar2 = hbar*hbar                                   #eV^2*s^2
    ans = c*c*hbar2*(3.*np.pi*np.pi*n)**(2./3.)/2./me   #cm^2 *eV^2 / eV / cm^2 = eV
    return ans  # units are eV
    
def lambdaTF(T,n):
    """
    $\lambda_{TF}^{-2} = 4 \pi e^2 n_e/\sqrt{T_e^2+(2/3E_F)^2}$ 
    """
    num = 4*np.pi*q2*n                   #eV / cm^2
    fermi = 2./3.*EF(n)                  #eV
    denom = np.sqrt(T*T + fermi*fermi)   #eV
    ansI2 = num/denom                    #1 / cm^2
    ans = np.sqrt(1./ansI2)              #cm 
    return ans   # units of cm
    
def kappa(T,n):
    """
    $\kappa = a_i/\lambda_e$
    """
    num = ai(n)
    denom = lambdaTF(T,n)
    ans = num/denom
    return ans  # this is dimensionless
    
def lambdaI(T,n,Z):
    """
    $\lambda_i = (T/4 \pi Z_i^2 e^2 n_i)^{1/2}$
    """
    denom = 4.*np.pi*Z*Z*q2*n            #eV/cm^2
    ans2 = T/denom                       #cm^2
    ans = np.sqrt(ans2)                  #cm
    return ans   # units are cm
    
def lambdaIon(lambdais):
    """
    $\lambda_{ion} = (\Sigma_i 1/\lambda_i^2)^{-1/2}$
    """
    s = len(lambdais)
    if s < 2:
        print('you only have one ion species so call lambdaI')
        return
    temp = 0.
    for i in range(s):
        temp += 1./lambdais[i]/lambdais[i]
    ans = np.sqrt(1./temp)
    return ans   # units in cm
    
def lambdaTot(le,li,gam):
    """
    $\lambda_{tot}=(1/\lambda_e^2 + 1/\lambda_{ion}^2(1/(1+3\Gamma)))^{-1/2}$.
    """
    add = 1./le/le + 1./li/li*(1./(1.+3*gam))
    ans = np.sqrt(1./add)
    return ans    # units in cm

def lambdaTots(le,li1,gami1,li2,gami2):
    """
    For 2 ions
    $\lambda_{tot}=(1/\lambda_e^2 + 1/\lambda_{ion}^2(1/(1+3\Gamma)))^{-1/2}$.
    """
    add = 1./le/le + 1./li1/li1*(1./(1.+3*gami1)) + 1./li2/li2*(1./(1.+3*gami2))
    ans = np.sqrt(1./add)
    return ans    # units in cm
    
def gi(k,gam):
    """
    $g_i=Z^2e^2/\lambda_{eff}T = \Gamma(\kappa^2+3\Gamma/(1+3\Gamma))^{1/2}$
    """
    denom = 1.+3*gam
    ans = gam*np.sqrt(k*k + 3*gam/denom)
    return ans  # this is a dimensionless parameter
    
def gij(Z1,Z2,lt,T):
    """
    $g_{ij} = Z_i Z_j e^2/\lambda T$.
    """
    num = Z1*Z2*q2        #eV cm
    denom = lt*T          #eV cm
    ans = num/denom       #unitless
    return ans  # dimensionless units
    
def muij(m1, m2):
    """
    Calculate the ion reduced mass value.
    """
    num = m1*m2
    denom = m1+m2
    ans = num/denom
    return ans # units in grams
    
def zbar(Z, AM, rho, T):
    """
    Finite Temperature Thomas Fermi Charge State using 
    R.M. More, "Pressure Ionization, Resonances, and the
    Continuity of Bound and Free States", Adv. in Atomic 
    Mol. Phys., Vol. 21, p. 332 (Table IV).
    
    Z = atomic number
    AM = atomic mass
    rho = density (g/cc)
    T = temperature (eV)
    """

    alpha = 14.3139
    beta = 0.6624
    a1 = 0.003323
    a2 = 0.9718
    a3 = 9.26148e-5
    a4 = 3.10165
    b0 = -1.7630
    b1 = 1.43175
    b2 = 0.31546
    c1 = -0.366667
    c2 = 0.983333
    
    R = rho/(Z*AM)
    T0 = T/Z**(4./3.)
    Tf = T0/(1+T0)
    A = a1*T0**a2+a3*T0**a4
    B = -np.exp(b0+b1*Tf+b2*Tf**7)
    C = c1*Tf+c2
    Q1 = A*R**B
    Q = (R**C+Q1**C)**(1/C)
    x = alpha*Q**beta

    return Z*x/(1 + x + np.sqrt(1 + 2.*x))

def zbar_n(ni, Z, T):
    """
    Finite Temperature Thomas Fermi Charge State using 
    R.M. More, "Pressure Ionization, Resonances, and the
    Continuity of Bound and Free States", Adv. in Atomic 
    Mol. Phys., Vol. 21, p. 332 (Table IV).
    
    Z = atomic number
    ni = number density in 1/cc
    T = temperature (eV)
    """
    
    NA = 6.02214e23
    alpha = 14.3139
    beta = 0.6624
    a1 = 0.003323
    a2 = 0.9718
    a3 = 9.26148e-5
    a4 = 3.10165
    b0 = -1.7630
    b1 = 1.43175
    b2 = 0.31546
    c1 = -0.366667
    c2 = 0.983333
    
    R = ni/(Z*NA)
    T0 = T/Z**(4./3.)
    Tf = T0/(1+T0)
    A = a1*T0**a2+a3*T0**a4
    B = -np.exp(b0+b1*Tf+b2*Tf**7)
    C = c1*Tf+c2
    Q1 = A*R**B
    Q = (R**C+Q1**C)**(1/C)
    x = alpha*Q**beta

    return Z*x/(1 + x + np.sqrt(1 + 2.*x))
    
def wt(m1,m2,lt,Z1,Z2,v):
    """
    This function returns the transformed form of omega for further evaluation in
    developing transport coefficents.
    """
    mu = muij(m1,m2)
    w2 = mu*lt*v*v/2./Z1/Z2/q2
    w  = np.sqrt(w2)
    return w
    
def phin(n,w):
    """
    We first must transform to scaled variables, $r \rightarrow \lambda r$,
    $\rho \rightarrow b/\lambda$, and $\omega^2 \rightarrow \mu \lambda v^2
    / 2 Z_i Z_j e^2$. Start with equation C7, $\sigma_{ij}^{(n)}(\omega, 
    \lambda) = 2 \pi \lambda^2 \phi_n(\omega)$.  First we look to equation
    C15 and definitions from there to get $\phi_n(\omega)$.Fitting form of
    phi_n equation C15 from PRE 93, 043203, Stanton and Murillo.  
    Assumes that w is in the tranformed variable form already.
    """
    if n==1:
        c0=0.30031
        c1=-0.69161
        c2=0.59607
        c3=-0.39822
        c4=-0.20685
        d0=0.48516
        d1=1.66045
        d2=-0.88687
        d3=0.55990
        d4=1.65798
        d5=-1.02457
    elif n==2:
        c0=0.40688
        c1=-0.86425
        c2=0.77461
        c3=-0.34471
        c4=-0.27626
        d0=0.83061
        d1=1.05229
        d2=-0.59902
        d3=1.41500
        d4=0.78874
        d5=-0.48155
    else:
        print('Error, n must equal 1 or 2 for this fit')
        return
    if w<=1.:    # C16 is solved
        num = c0+c1*np.log(w)+c2*np.log(w)**2.0+c3*np.log(w)**3.0
        denom = 1.0+c4*np.log(w)                                  
        ans = num/denom
        return ans
    else:       # C17 and C18 are solved
        num = d0+d1*np.log(w)+d2*np.log(w)**2.0+np.log(w)**3.0
        denom = d3+d4*np.log(w)+d5*np.log(w)**2.0+np.log(w)**3.0
        P=num/denom
        ans = n*np.log(1+w*w)*P/2.0/w/w/w/w
    return ans
        
def phiIJ(n,w,lt):
    """
    Solve for the momentum transfer cross section equation C7.  Assumes w is 
    in transformed form, see appendix C.
    """
    temp = phin(n,w)
    ans = 2.0*np.pi*lt*lt*temp
    return ans
    
def omegaij(n,m,Z1,Z2,T,lt,mu):
    """
    Solve for the collision integral of equation C19.  This solution is based on a fit to K_nm
    based on the value of the couping parameter g. These fits for the collision integral
    will be divided into two parts based on the coupling parameter 
    $g_{ij}=Z_1 Z_2 e^2/\lambda T$.  These integrals depend on two parameters (n,m) 
    and we will give results for (1,1), (1,2), (1,3), and (2,2).  The integral is given 
    as equation C19 by $\Omega_{ij}^{(n,m)} = \sqrt{2 \pi/\mu_{ij}} 
    (Z_i Z_j e^2)^2/T^{3/2} K_{nm}(g)$.  $K_{nm}$ is fit based on $g < 1$ and $g \ge 1$ 
    and is given as equations C22-C24.
    """
    if (n==1 and m==1):
        a1 = 1.4660
        a2 = -1.7836
        a3 = 1.4313
        a4 = -0.55833
        a5 = 0.061162
        b0 = 0.081033
        b1 = -0.091336
        b2 = 0.051760
        b3 = -0.50026
        b4 = 0.17044
    elif (n==1 and m==2):
        a1 = 0.52094
        a2 = 0.25153
        a3 = -1.1337
        a4 = 1.2155
        a5 = -0.43784
        b0 = 0.20572
        b1 = -0.16536
        b2 = 0.061572
        b3 = -0.12770
        b4 = 0.066993
    elif (n==1 and m==3):
        a1 = 0.30346
        a2 = 0.23739
        a3 = -0.62167
        a4 = 0.56110
        a5 = -0.18046
        b0 = 0.68375
        b1 = -0.38459
        b2 = 0.10711
        b3 = 0.10649
        b4 = 0.028760
    elif (n==2 and m==2):
        a1 = 0.85401
        a2 = -0.22898
        a3 = -0.60059
        a4 = 0.80591
        a5 = -0.30555
        b0 = 0.43475
        b1 = -0.21147
        b2 = 0.11116
        b3 = 0.19665
        b4 = 0.15195
    else:
        print('(n,m) pair is not valid for this fit.')
        
    g = gij(Z1,Z2,lt,T)
    mn1 = m-1
    if g <= 1:
        fac = -(n/4.)*factorial(mn1)
        ser = a1*g + a2*g*g + a3*g*g*g + a4*g*g*g*g + a5*g*g*g*g*g
        knm = fac*np.log(ser)
    else:
        num = b0 + b1*np.log(g) + b2*np.log(g)*np.log(g)
        denom = 1 + b3*g + b4*g*g
        knm = num/denom
    
    fac = np.sqrt(2.*np.pi/mu)       #1/g^1/2
    fac1 = Z1*Z2*q2                  #eV*cm
    fac2 = fac1*fac1                 #eV^2*cm^2   
    ans = fac*fac2/(T)**(1.5)*knm    #eV^1/2 cm^2 / g^1/2
    ####CONVERT TO CGS
    ans = ans / np.sqrt(erg_to_ev)   #eV^1/2 cm^2 / g^1/2 * (g^1/2 cm / s) / eV = cm^3 / s
    return ans                       #cm^3 / s
    
def diffusion(T,n,m,omega11):
    """
    This fucntion solves equation B5 from Stanton-Murillo for the self diffusion coefficient.
    """
    num = 3.*T                       #eV
    denom = 8.*n*m*omega11           #g / s
    ans = num/denom                  #eV s / g
    #Convert to CGS
    ans = ans / erg_to_ev            #s / g * g cm^2 / s^2 = cm^2 / s
    return ans                       #cm^2 / s 
    
def viscocity(T,omega22):
    """
    This function calculates the viscocity for a single species given by equation B6 of
    Stanton-Murillo.
    """
    num = 5.*T                       #eV
    denom = 8.*omega22               #cm^3 / s
    ans = num/denom                  #eV s / cm^3
    #convert to cgs
    ans = ans / erg_to_ev            #s / cm^3 * g cm^2 / s^2 = g / cm s
    return ans                       #g / cm s
    
def conduction(T,m,omega22):
    """
    This function returns the thermal conduction coefficient given by equation B7 of
    Stanton Murrilo.
    """
    num = 75.*T                      #eV
    denom = 32.*m*omega22            #g * cm^3 / s
    ans = num/denom                  #eV * s / g cm^3
    #convert to cgs
    ans = ans / erg_to_ev            #s / g cm^3 * g cm^2 / s^2 = 1 / cm s
    return ans                       # 1 / cm s

