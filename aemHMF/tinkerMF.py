"""
This file contains the actual Tinker08 mass function. It uses emu to get the mass function parameters.
"""
from classy import Class
from cluster_toolkit import bias
import numpy as np
from scipy import special, integrate
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
from scipy.interpolate import RectBivariateSpline as RBS
#from scipy.interpolate import interp2d as RBS
import emu

#Physical constants
G = 4.51715e-48 #Newton's gravitional constant in Mpc^3/s^2/Solar Mass
Mpcperkm = 3.240779289664256e-20 #Mpc/km; used to convert H0 to s^-1
rhocrit = 3.*(Mpcperkm*100)**2/(8*np.pi*G) #Msun h^2/Mpc^3

class tinkerMF(object):
    
    def __init__(self):
        self.t_08 = emu.emu()

    def set_cosmology(self, cosmo_dict):
        self.cosmo_dict = cosmo_dict
        self.rhom = cosmo_dict['Omega_m']*rhocrit #Msun h^2/Mpc^3
        params = {
            'output': 'mPk', #linear only
            'h': cosmo_dict['h'],
            #'A_s': cosmo_dict['A_s'],
            'sigma8': cosmo_dict['sigma8'],
            'n_s': cosmo_dict['n_s'],
            'w0_fld': cosmo_dict['w0'],
            'wa_fld': 0.0,
            'Omega_b': cosmo_dict['Omega_b'],
            'Omega_cdm': cosmo_dict['Omega_m'] - cosmo_dict['Omega_b'],
            'Omega_Lambda': 1.- cosmo_dict['Omega_m'],
            'N_eff':cosmo_dict['N_eff'],
            'P_k_max_1/Mpc':10.,
            'z_max_pk':10.
        }
        self.classcosmo = Class()
        self.classcosmo.set(params)
        self.classcosmo.compute()
        Nm = 1000
        Nz = 20
        M = np.logspace(11, 17, num=Nm) #Msun/h
        R = self.MtoR(M) #Mpc already
        z = np.linspace(0, 3, num=Nz)
        sigmas = np.zeros((Nz, Nm))
        for i in range(Nz):
            h = self.cosmo_dict['h']
            k = np.logspace(-5, 1, num=1000) #h Mpc^-1
            p = np.array([self.classcosmo.pk_lin(ki, z[i]) for ki in k])*h**3 #Mpc^3/h^3
            sigmas[i] = np.sqrt(bias.sigma2_at_M(M, k/h, p, self.cosmo_dict['Omega_m']))
        #self.lnsigma_spl = RBS(z, np.log(M), np.log(sigmas.T)) #If using interp2d
        self.lnsigma_spl = RBS(z, np.log(M), np.log(sigmas)) #If using RBS for real
        cos = self.cos_from_dict(cosmo_dict)
        self.t08_slopes_intercepts = self.t_08.predict_slopes_intercepts(cos)
        
    def cos_from_dict(self, cosmo_dict):
        cd = cosmo_dict
        om = cd['Omega_m']
        ob = cd['Omega_b']
        w0 = cd['w0']
        ns = cd['n_s']
        h  = cd['h']
        Neff = cd['N_eff']
        #As = cd['As']
        s8 = cd['sigma8']
        H0 = h*100
        Obh2 = ob*h*h
        Och2 = (om-ob)*h*h
        return np.array([Obh2, Och2, w0, ns, H0, Neff, s8])
        #return np.array([Obh2, Och2, w0, ns, As, H0, Neff])

    def calc_normalization(self):
        #Calculates B, the normalization of the T08 MF.
        d,e,f,g = self.t08_params
        gamma_d2 = special.gamma(d*0.5)
        gamma_f2 = special.gamma(f*0.5)
        gnd2 = g**(-d*0.5)
        gnf2 = g**(-f*0.5)
        ed = e**d
        self.B = 2.0/(ed * gnd2 * gamma_d2 + gnf2 * gamma_f2)
        return
        
    def merge_t08_params(self, z):
        k = 1./(1.+z)-0.5
        d0,d1,f0,f1,g0,g1 = self.t08_slopes_intercepts
        d = d0 + k*d1
        e = np.array([1.0]) #Default Tinker08 value
        f = f0 + k*f1
        g = g0 + k*g1
        #a = 1./(1.+z)
        #print "in merge",a, d ,e, f, g
        self.t08_params = np.array([d, e, f, g]).flatten()
        return

    def MtoR(self, M):
        return (M/(4./3.*np.pi*self.rhom))**(1./3.)/self.cosmo_dict['h'] #Lagrangian radius in Mpc

    def sigmaM(self, M, z):
        #return np.exp(self.sigma_spl(z, np.log(M))[0]) #If using interp2d
        return np.exp(self.lnsigma_spl(z, np.log(M))[0][0]) #If using RBS for real
        
    def Gsigma(self, sigma, z):
        if not hasattr(self, "redshift"):
            self.redshift = z
            self.merge_t08_params(z)
            self.calc_normalization() #Recalculate B
        if self.redshift != z:
            self.redshift = z
            self.merge_t08_params(z)
            self.calc_normalization() #Recalculate B
        d,e,f,g = self.t08_params
        return self.B*((sigma/e)**-d + sigma**-f) * np.exp(-g/sigma**2)

    def GM(self, M, z):
        if not hasattr(self, "redshift"):
            self.redshift = z
            self.merge_t08_params(z)
            self.calc_normalization() #Recalculate B
        if self.redshift != z:
            self.redshift = z
            self.merge_t08_params(z)
            self.calc_normalization() #Recalculate B
        sigma = self.sigmaM(M, z)
        return self.Gsigma(sigma, z)

    def dndlM(self, M, z):
        gsigma = self.GM(M, z)
        dM = 1e-8*M
        dlnsiginvdm = (self.lnsigma_spl(z, np.log(M-dM*0.5))[0,0] - self.lnsigma_spl(z, np.log(M+dM*0.5))[0][0])/dM
        return gsigma * self.rhom * dlnsiginvdm

    def n_bin(self, Mlow, Mhigh, z):
        M = np.logspace(np.log10(Mlow), np.log10(Mhigh), num=2000)
        lM = np.log(M)
        dndlM = np.array([self.dndlM(Mi, z) for Mi in M])
        spl = IUS(lM, dndlM)
        return spl.integral(np.log(Mlow), np.log(Mhigh))

    def n_bins(self, Mbins, z):
        return np.array([self.n_bin(Mbi[0], Mbi[1], z) for Mbi in Mbins])

    def peak_height(self, M, z):
        return 1.686/self.sigmaM(M, z)

if __name__ == "__main__":
    cos = {"Omega_m":0.3,
           "Omega_b":0.05,
           "h":0.7,
           "sigma8":0.8,
           "n_s":0.96,
           "w0":-1.0,
           "N_eff":3.0146}
    n = tinkerMF()
    n.set_cosmology(cos)
    M = np.logspace(12, 16, num=100)
    import matplotlib.pyplot as plt
    for i in xrange(0,4):
        z = float(i)
        dndlM = np.array([n.dndlM(Mi, z) for Mi in M])
        plt.loglog(M, dndlM, label="z=%d"%i, ls='-')
    plt.legend(loc=0)
    plt.xlabel("Mass")
    plt.xscale('log'); plt.show()
