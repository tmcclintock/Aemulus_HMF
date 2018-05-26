"""
This file contains the actual Tinker08 mass function. It uses emu to get the mass function parameters.
"""
from classy import Class
from cluster_toolkit import massfunction
from cluster_toolkit import peak_height as ph
import numpy as np
from scipy import special, integrate
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
from scipy.interpolate import RectBivariateSpline as RBS
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
        h = cosmo_dict["H0"]/100.
        Omega_m = (cosmo_dict["Obh2"]+cosmo_dict["Och2"])/h**2
        self.rhom = Omega_m*rhocrit #Msun h^2/Mpc^3
        self.h = h
        self.Omega_m = Omega_m
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
            'z_max_pk':5.03
        }
        self.cc = Class()
        self.cc.set(params)
        self.cc.compute()
        self.k = np.logspace(-5, 1, num=1000) #Mpc^-1
        self.M_array = np.logspace(11, 16.5, num=250)
        self.t08_slopes_intercepts = self.params_emu.predict_slopes_intercepts(self.cos_from_dict(cosmo_dict))
        self.sig  = {}
        self.sigt = {}
        self.sigb = {}
        self.ccp  = {}

    def cos_from_dict(self, cd):
        Och2 = cd['Och2']
        Obh2 = cd['Obh2']
        w0 = cd['w0']
        ns = cd['n_s']
        H0  = cd['H0']
        Neff = cd['N_eff']
        l10As = cd['ln10^{10}A_s']
        return np.array([Obh2, Och2, w0, ns, l10As, H0, Neff])

    def get_tinker_parameters(self, z):
        params = self.t08_slopes_intercepts
        #d0,e0,f0,g0,d1,f1,g1 = params
        #e1 = 0.24327712
        #d0,e0,f0,g0,d1,e1,g1 = params
        #f1 = 0.11628991
        e0,f0,g0,d1,e1,g1 = params
        d0, f1 = [ 2.39279115, 0.11628991]
        """
        if len(params) == 8:
            d0,d1,e0,e1,f0,f1,g0,g1 = params
        elif len(params) == 6:
            d0,d1,f0,f1,g0,g1 = params
            e0,e1 = 1.0, 0.0
        elif len(params) == 7:
            name = self.params_emu.name
            if 'e0' in name:
                d0,d1,e0,f0,f1,g0,g1 = params
                e1 = 0.3098
            elif 'e1' in name:
                d0,d1,e1,f0,f1,g0,g1 = params
                e0 = 1.11
        """
        x = 1./(1+z)-0.5
        d = d0 + x*d1
        e = e0 + x*e1
        f = f0 + x*f1
        g = g0 + x*g1
        return  np.array([d,e,f,g]).flatten()

    def _add_sigma_at_z(self, z):
        h = self.h
        Omega_m = self.Omega_m
        k = self.k #Mpc^-1
        p = np.array([self.cc.pk_lin(ki, z) for ki in k])*h**3 #[Mpc/h]^3
        M = self.M_array
        Mt = M*(1-1e-6*0.5)
        Mb = M*(1+1e-6*0.5)
        self.sig[z] = ph.sigma2_at_M(M, k/h, p, Omega_m)
        self.sigt[z] = ph.sigma2_at_M(Mt, k/h, p, Omega_m)
        self.sigb[z] = ph.sigma2_at_M(Mb, k/h, p, Omega_m)
        self.ccp[z] = p
        
    def GM(self, M, z):
        h = self.h
        Omega_m = self.Omega_m
        k = self.k #Mpc^-1
        p = np.array([self.cc.pk_lin(ki, z) for ki in k])*h**3 #[Mpc/h]^3
        d, e, f, g = self.get_tinker_parameters(z)
        return massfunction.G_at_M(M, k/h, p, Omega_m, d, e, f, g)

    def Gsigma(self, sigma, z):
        d, e, f, g = self.get_tinker_parameters(z)
        return massfunction.G_at_sigma(sigma, d, e, f, g)

    def _M_and_dndM(self, z):
        Omega_m = self.Omega_m
        d, e, f, g = self.get_tinker_parameters(z)
        if z not in self.sig:
            self._add_sigma_at_z(z)
        sig = self.sig[z]
        sigt = self.sigt[z]
        sigb = self.sigb[z]
        M = self.M_array
        dndM = massfunction._dndM_sigma2_precomputed(M, sig, sigt, sigb, Omega_m, d, e, f, g)
        return M, dndM
    
    def dndlM(self, M, z):
        h = self.cosmo_dict['H0']/100.
        Omega_m = (self.cosmo_dict['Obh2']+self.cosmo_dict['Och2'])/h**2
        k = self.k #Mpc^-1
        if z not in self.ccp:
            self._add_sigma_at_z(z)
        p = self.ccp[z]
        d, e, f, g = self.get_tinker_parameters(z)
        return massfunction.dndM_at_M(M, k/h, p, Omega_m, d, e, f, g)*M

    def n_in_bin(self, Mlow, Mhigh, z):
        M, dndM = self._M_and_dndM(z)
        out =  massfunction.n_in_bin(Mlow, Mhigh, M, dndM)
        return out
        
    def n_in_bins(self, Mbins, z):
        return np.array([self.n_in_bin(Mbi[0], Mbi[1], z) for Mbi in Mbins])
                
    def peak_height(self, M, z):
        h = self.h
        Omega_m = self.Omega_m
        k = self.k #Mpc^-1
        if z not in self.ccp:
            self._add_sigma_at_z(z)
        p = self.ccp[z]
        return 1.686/np.sqrt(ph.sigma2_at_M(M, k/h, p, Omega_m))

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
    print n.get_tinker_parameters(0)
    M = np.logspace(12, 16, num=20)
    Mbins = np.array([M[:-1], M[1:]]).T
    import matplotlib.pyplot as plt
    z = 0.0
    dndlM = n.dndlM(M, z)
    n = n.n_in_bins(Mbins, z)
    plt.loglog(M, dndlM, label="z=%.1f"%z, ls='-')
    plt.loglog(np.mean(Mbins,1), n)
    plt.legend(loc=0)
    plt.xlabel("Mass")
    plt.xscale('log')
    plt.show()
