"""
This file contains the actual Tinker08 mass function. It uses t08_emu to get the mass function parameters.
"""
#Public modules
import cosmocalc as cc
import numpy as np
from scipy import special, integrate
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
import t08_emu

#Physical constants
G = 4.51701e-48 #Newton's gravitional constant in Mpc^3/s^2/Solar Mass
Mpcperkm = 3.24077927001e-20 #Mpc/km; used to convert H0 to s^-1
rhocrit = 3.*(Mpcperkm*100)**2/(8*np.pi*G) #Msun h^2/Mpc^3

class n_t08(object):
    
    def __init__(self, cosmo_dict, a=1.0):
        self.cosmo_dict = cosmo_dict
        self.a = a
        self.rhom       = cosmo_dict['om']*rhocrit #Msun h^2/Mpc^3
        cc.set_cosmology(cosmo_dict)
        t = t08_emu.t08_emu()
        cos = self.cos_from_dict()
        self.t08_slopes_intercepts = t.predict_slopes_intercepts(cos)
        self.merge_t08_params(a)
        self.calc_normalization()

    def cos_from_dict(self):
        cd = self.cosmo_dict
        om = cd['om']
        ob = cd['ob']
        w0 = cd['w0']
        ns = cd['ns']
        h  = cd['h']
        Neff = cd['Neff']
        s8 = cd['s8']
        H0 = h*100
        Obh2 = ob*h*h
        Och2 = (om-ob)*h*h
        return np.array([Obh2, Och2, w0, ns, H0, Neff, s8])

    def calc_normalization(self):
        #Calculates B, the normalization of the T08 MF.
        d,e,f,g = self.t08_params
        gamma_d2 = special.gamma(d*0.5)
        gamma_f2 = special.gamma(f*0.5)
        log_g = np.log(g)
        gnd2 = g**(-d*0.5)
        gnf2 = g**(-f*0.5)
        ed = e**d
        self.B = 2.0/(ed * gnd2 * gamma_d2 + gnf2 * gamma_f2)
        return
        
    def merge_t08_params(self, a):
        k = a-0.5
        d0,d1,f0,f1,g0,g1 = self.t08_slopes_intercepts
        d = d0 + k*d1
        e = np.array([1.0]) #Default Tinker08 value
        f = f0 + k*f1
        g = g0 + k*g1
        self.t08_params = np.array([d, e, f, g]).flatten()
        return

    def dndlM(self, M, a):
        if self.a != a: self.merge_t08_params(a)
        sM = cc.sigmaMtophat(M, a)
        d,e,f,g = self.t08_params
        gsigma = self.B*((sM/e)**-d + sM**-f) * np.exp(-g/sM**2)
        dM = 1e-6*M
        dlnsiginvdm = np.log(cc.sigmaMtophat(M-dM/2, a)/cc.sigmaMtophat(M+dM/2, a))/dM
        return gsigma * self.rhom * dlnsiginvdm

    def n_bin(self, Mlow, Mhigh, a):
        M = np.logspace(np.log10(Mlow), np.log10(Mhigh), num=100, base=10)
        lM = np.log(M)
        dndlM = np.array([self.dndlM(Mi, a) for Mi in M])
        spl = IUS(lM, dndlM)
        return spl.integral(np.log(Mlow), np.log(Mhigh))

if __name__ == "__main__":
    cd = {"om":0.3,"ob":0.05,"ol":1.-0.3,"ok":0.0,"h":0.7,"s8":0.77,"ns":0.96,"w0":-1.0,"wa":0.0,"Neff":3.0}
    n = n_t08(cd)
    V = 1050.**3
    M = np.logspace(12, 16, num=100, base=10)

    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("brg")
    for i in range(3):
        a = 1./(1.+i)
        c = (1-a)/2
        dndlM = np.array([n.dndlM(Mi, a) for Mi in M])
        plt.loglog(M, dndlM, c=cmap(c))
    plt.ylim(1e-12, 1e-1)
    plt.show()
    plt.clf()

    M = np.logspace(11, 16, num=11, base=10)
    Mbins = np.array([M[:-1], M[1:]]).T
    Mave = np.mean(Mbins, 1)
    for i in range(3):
        a = 1./(1.+i)
        c = (1-a)/2
        number = np.array([n.n_bin(Mbi[0], Mbi[1], a) for Mbi in Mbins])
        plt.loglog(Mave, number*V, c=cmap(c))
    plt.ylim(1e0, 1e8)
    plt.show()
