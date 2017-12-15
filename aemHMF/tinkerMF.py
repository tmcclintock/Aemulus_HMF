"""
This file contains the actual Tinker08 mass function. It uses emu to get the mass function parameters.
"""
from classy import Class
from cluster_toolkit import massfunction, bias
import numpy as np
from scipy import special, integrate
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
from scipy.interpolate import RectBivariateSpline as RBS
#from scipy.interpolate import interp2d as RBS
import emu, sys

#Physical constants
G = 4.51715e-48 #Newton's gravitional constant in Mpc^3/s^2/Solar Mass
Mpcperkm = 3.240779289664256e-20 #Mpc/km; used to convert H0 to s^-1
rhocrit = 3.*(Mpcperkm*100)**2/(8*np.pi*G) #Msun h^2/Mpc^3

class tinkerMF(object):
    
    def __init__(self):
        self.params_emu = emu.emu()

    def set_cosmology(self, cosmo_dict):
        self.cosmo_dict = cosmo_dict
        Omh2 = cosmo_dict["Obh2"]+cosmo_dict["Och2"]
        h = cosmo_dict["H0"]/100.
        Omega_m = Omh2/h**2
        self.rhom = Omega_m*rhocrit #Msun h^2/Mpc^3
        params = {
            'output': 'mPk', #linear only
            'H0': cosmo_dict['H0'],
            'ln10^{10}A_s': cosmo_dict['ln10^{10}A_s'],
            'n_s': cosmo_dict['n_s'],
            'w0_fld': cosmo_dict['w0'],
            'wa_fld': 0.0,
            'omega_b': cosmo_dict['Obh2'],
            'omega_cdm': cosmo_dict['Och2'],
            'Omega_Lambda': 1.-Omega_m,
            'N_eff':cosmo_dict['N_eff'],
            'P_k_max_1/Mpc':10.,
            'z_max_pk':10.
        }
        self.classcosmo = Class()
        self.classcosmo.set(params)
        self.classcosmo.compute()
        self.k_array = np.logspace(-5, 1, num=1000) #Mpc^-1
        cos = self.cos_from_dict(cosmo_dict)
        self.t08_slopes_intercepts = self.params_emu.predict_slopes_intercepts(cos)
        
    def cos_from_dict(self, cosmo_dict):
        cd = cosmo_dict
        Och2 = cd['Och2']
        Obh2 = cd['Obh2']
        w0 = cd['w0']
        ns = cd['n_s']
        H0  = cd['H0']
        Neff = cd['N_eff']
        l10As = cd['ln10^{10}A_s']
        out = np.array([Obh2, Och2, w0, ns, l10As, H0, Neff])
        return out

    def get_tinker_parameters(self, z):
        d0,d1,e0,e1,f0,f1,g0,g1 = self.t08_slopes_intercepts
        x = (1.+z)-0.5
        d = d0 + x*d1
        e = e0 + x*e1
        f = f0 + x*f1
        g = g0 + x*g1
        return  np.array([d,e,f,g]).flatten()

    def GM(self, M, z):
        h = self.cosmo_dict['H0']/100.
        Omega_m = (self.cosmo_dict['Obh2']+self.cosmo_dict['Och2'])/h**2
        k = self.k_array #Mpc^-1
        p = np.array([self.classcosmo.pk_lin(ki, z) for ki in k])*h**3 #[Mpc/h]^3
        d, e, f, g = self.get_tinker_parameters(z)
        return massfunction.G_at_M(M, k/h, p, Omega_m, d, e, f, g)

    def Gsigma(self, sigma, z):
        d, e, f, g = self.get_tinker_parameters(z)
        return massfunction.G_at_sigma(sigma, d, e, f, g)
        
    def dndlM(self, M, z):
        h = self.cosmo_dict['H0']/100.
        Omega_m = (self.cosmo_dict['Obh2']+self.cosmo_dict['Och2'])/h**2
        k = self.k_array #Mpc^-1
        p = np.array([self.classcosmo.pk_lin(ki, z) for ki in k])*h**3 #[Mpc/h]^3
        d, e, f, g = self.get_tinker_parameters(z)
        return massfunction.dndM_at_M(M, k/h, p, Omega_m, d, e, f, g)*M

    def n_bin(self, Mlow, Mhigh, z):
        M = np.logspace(np.log10(Mlow), np.log10(Mhigh), num=100)
        dndM = self.dndlM(M, z)/M
        return massfunction.n_in_bin(Mlow, Mhigh, M, dndM)
        
    def n_bins(self, Mbins, z):
        return np.array([self.n_bin(Mbi[0], Mbi[1], z) for Mbi in Mbins])
    
    def peak_height(self, M, z):
        h = self.cosmo_dict['H0']/100.
        Omega_m = (self.cosmo_dict['Obh2']+self.cosmo_dict['Och2'])/h**2
        k = self.k_array #Mpc^-1
        p = np.array([self.classcosmo.pk_lin(ki, z) for ki in k])*h**3 #[Mpc/h]^3
        return 1.686/np.sqrt(bias.sigma2_at_M(M, k, p, Omega_m))

if __name__ == "__main__":
    cos = {"Och2":0.12,
           "Obh2":0.023,
           "H0":70.,
           "ln10^{10}A_s":3.093,
           "n_s":0.96,
           "w0":-1.0,
           "N_eff":3.0146}
    n = tinkerMF()
    n.set_cosmology(cos)
    M = np.logspace(12, 16, num=20)
    Mbins = np.array([M[:-1], M[1:]]).T
    import matplotlib.pyplot as plt
    for i in xrange(0,1):
        z = float(i)
        dndlM = n.dndlM(M, z)
        plt.loglog(M, dndlM, label="z=%d"%i, ls='-')
    plt.legend(loc=0)
    plt.xlabel("Mass")
    plt.xscale('log')
    plt.show()
