"""
This contains the Aemulus HMF emulator.
"""
from __future__ import absolute_import, division, print_function

import os, sys
from aemHMF import tinkerMF
from aemHMF import residual_gp
import numpy as np

class Aemulus_HMF(object):

    #This is the format the cosmological dictionary should take.
    defaul_cosmology = {"om":0.3,"ob":0.05,"ol":1.-0.3,"ok":0.0,"h":0.7,"s8":0.77,"ns":0.96,"w0":-1.0,"Neff":3.0}# "wa":0.0 is assumed internally

    def __init__(self):
        self.tinkerMF = None
        self.residualgp = residual_gp.residual_gp()

    def set_cosmology(self, cosmo_dict):
        cosmo_dict["wa"] = 0.0 #Aemulus simulations don't have wa
        self.tinkerMF = tinkerMF.tinkerMF()
        self.tinkerMF.set_cosmology(cosmo_dict)

    def set_default_cosmology(self):
        self.tinkerMF = tinkerMF.tinkerMF()
        self.tinkerMF.set_cosmology(self.defaul_cosmology)

    def Mtosigma(self, M, a):
        if type(M) is list or type(M) is np.ndarray:
            return np.array([self.tinkerMF.Mtosigma(Mi, a) for Mi in M])
        else:
            return self.tinkerMF.Mtosigma(M, a)

    def multiplicity(self, M, a):
        if type(M) is list or type(M) is np.ndarray:
            return np.array([self.tinkerMF.GM(Mi, a) for Mi in M])
        else:
            return self.tinkerMF.GM(M, a)

    def multiplicity_sigma(self, sigma, a):
        if type(sigma) is list or type(sigma) is np.ndarray:
            return np.array([self.tinkerMF.Gsigma(sigmai, a) for sigmai in sigma])
        else:
            return self.tinkerMF.Gsigma(sigma, a)


    def dndlM(self, M, a):
        if type(M) is list or type(M) is np.ndarray:
            return np.array([self.tinkerMF.dndlM(Mi, a) for Mi in M])
        else:        
            return self.tinkerMF.dndlM(M, a)

    def n_bin(self, Mlow, Mhigh, a):
        return self.tinkerMF.n_bin(Mlow, Mhigh, a)

    def n_bins(self, Mbins, a):
        return self.tinkerMF.n_bins(Mbins, a)

    def residual_realization(self, M, a, Nrealizations=1):
        z = 1-1./a
        nu = np.array([self.tinkerMF.peak_height(Mi, a) for Mi in M])
        return np.array([self.residualgp.residual_realization(nu, np.ones_like(nu)*z) for i in range(Nrealizations)])

if __name__ == "__main__":
    a = 1.0 #Scale factor
    hmf = Aemulus_HMF()
    hmf.set_default_cosmology()
    Medges = np.logspace(11, 16, num=31)
    Mbins = np.array([Medges[:-1], Medges[1:]]).T
    M = np.mean(Mbins, 1)
    n = hmf.n_bins(Mbins, a)
    fs = hmf.residual_realization(M, a, 50)

    import matplotlib.pyplot as plt
    fig, axarr = plt.subplots(2, sharex=True)
    axarr[0].loglog(M, n)
    axarr[0].set_ylabel('n(M) (Mpc**-3)')
    for i in range(len(fs)):
        ns = n*(1+fs[i])
        axarr[1].plot(M, (n-ns)/ns, alpha=0.2)
    axarr[1].set_xlabel(r'$M_{\rm halo}$')
    plt.show()
