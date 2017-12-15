"""
This contains the Aemulus HMF emulator.
"""
import os, sys
import tinkerMF
import residual_gp
import numpy as np

class Aemulus_HMF(object):

    #This is the format the cosmological dictionary should take.
    default_cosmology = {"Och2":0.1225,
                         "Obh2":0.0245,
                         "H0":70., #km/sec/Mpc
                         "ln10^{10}A_s":3.093,
                         "n_s":0.96,
                         "w0":-1.0,
                         "N_eff":3.0,
                         "wa":0.0} #This is a requirement

    def __init__(self):
        self.tinkerMF = None
        self.residualgp = residual_gp.residual_gp()

    def set_cosmology(self, cosmo_dict=None):
        if not cosmo_dict:
            print "No cosmology set. Using default cosmology."
            cosmo_dict = self.default_cosmology
        if 'wa' in cosmo_dict:
            if cosmo_dict['wa'] != 0:
                raise Exception("AemulusError: 'wa' must be set to 0")
        self.tinkerMF = tinkerMF.tinkerMF()
        cosmo_dict['wa'] = 0.0 #Aemulus simulations don't have wa
        self.tinkerMF.set_cosmology(cosmo_dict)

    def Mtosigma(self, M, z):
        return 1.686/self.tinkerMF.peak_height(M, z)

    def multiplicity(self, M, z):
        if type(M) is list or type(M) is np.ndarray:
            return np.array([self.tinkerMF.GM(Mi, z) for Mi in M])
        else:
            return self.tinkerMF.GM(M, z)

    def multiplicity_sigma(self, sigma, z):
        return self.tinkerMF.Gsigma(sigma, z)

    def dndlM(self, M, z):
        return self.tinkerMF.dndlM(M, z)

    def n_bin(self, Mlow, Mhigh, z):
        return self.tinkerMF.n_bin(Mlow, Mhigh, z)

    def n_bins(self, Mbins, z):
        return self.tinkerMF.n_bins(Mbins, z)

    def residual_realization(self, M, z, Nrealizations=1):
        nu = self.tinkerMF.peak_height(M, z)
        return np.array([self.residualgp.residual_realization(nu, np.ones_like(nu)*z) for i in range(Nrealizations)])

if __name__ == "__main__":
    hmf = Aemulus_HMF()
    hmf.set_cosmology()
    Medges = np.logspace(12, 16, num=11)
    z = 0
    Mbins = np.array([Medges[:-1], Medges[1:]]).T
    M = np.mean(Mbins, 1)
    n = hmf.n_bins(Mbins, z)
    fs = hmf.residual_realization(M, z, 50)
    print n
    
    def wrapper():
        n = hmf.n_bins(Mbins, z)
    import timeit
    print timeit.timeit(wrapper, number=10)
