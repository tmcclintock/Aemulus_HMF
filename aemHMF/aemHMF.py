"""
This contains the Aemulus HMF emulator.
"""
import os, sys
import n_t08
import residual_gp
from n_t08 import peak_height
import numpy as np

#This is the formal the cosmological dictionary should take.
cd = {"om":0.3,"ob":0.05,"ol":1.-0.3,"ok":0.0,"h":0.7,"s8":0.77,"ns":0.96,"w0":-1.0,"Neff":3.0}# "wa":0.0 is assumed internally


class Aemulus_HMF(object):
    
    def __init__(self):
        self.n_t08 = None
        self.residualgp = residual_gp.residual_gp()

    def set_cosmology(self, cosmo_dict):
        cosmo_dict["wa"] = 0.0 #Aemulus simulations don't have wa
        self.n_t08 = n_t08.n_t08()
        self.n_t08.set_cosmology(cosmo_dict)

    def set_default_cosmology(self):
        cd = {"om":0.3,"ob":0.05,"ol":1.-0.3,"ok":0.0,"h":0.7,"s8":0.77,"ns":0.96,"w0":-1.0,"wa":0.0,"Neff":3.0}
        self.n_t08 = n_t08.n_t08()
        self.n_t08.set_cosmology(cd)

    def Mtosigma(self, M, a):
        if type(M) is list or type(M) is np.ndarray:
            return np.array([self.n_t08.Mtosigma(Mi, a) for Mi in M])
        else:
            return self.n_t08.Mtosigma(M, a)

    def multiplicity(self, M, a):
        if type(M) is list or type(M) is np.ndarray:
            return np.array([self.n_t08.GM(Mi, a) for Mi in M])
        else:
            return self.n_t08.GM(M, a)

    def multiplicity_sigma(self, sigma, a):
        if type(sigma) is list or type(sigma) is np.ndarray:
            return np.array([self.n_t08.Gsigma(sigmai, a) for sigmai in sigma])
        else:
            return self.n_t08.Gsigma(sigma, a)


    def dndlM(self, M, a):
        if type(M) is list or type(M) is np.ndarray:
            return np.array([self.n_t08.dndlM(Mi, a) for Mi in M])
        else:        
            return self.n_t08.dndlM(M, a)

    def n_bin(self, Mlow, Mhigh, a):
        return self.n_t08.n_bin(Mlow, Mhigh, a)

    def n_bins(self, Mbins, a):
        return self.n_t08.n_bins(Mbins, a)

    def residual_realization(self, M, a, Nrealizations):
        z = 1-1./a
        nu = np.array([peak_height(Mi, a) for Mi in M])
        return np.array([self.residualgp.residual_realization(nu, np.ones_like(nu)*z) for i in range(Nrealizations)])

if __name__ == "__main__":
    a = 1.0 #Scale factor
    hmf = Aemulus_HMF()
    hmf.set_default_cosmology()
    Medges = np.logspace(11, 16, num=11)
    Mbins = np.array([Medges[:-1], Medges[1:]]).T
    M = np.mean(Mbins, 1)
    n = hmf.n_bins(Mbins, a)
    V = 1050.**3 #Mpc/h ^3
    fs = hmf.residual_realization(M, a, 50)

    import matplotlib.pyplot as plt
    fig, axarr = plt.subplots(2, sharex=True)
    axarr[0].loglog(M, n*V, c='k', label=r"$N_{\rm emu}$")
    axarr[0].legend(loc=0, fontsize=14, frameon=False)
    
    for i in range(len(fs)):
        ns = n*(1+fs[i])
        axarr[1].plot(M, (n-ns)/ns, c='blue', alpha=0.2, zorder=-1)
        
    axarr[1].axhline(0, c='k', ls='-')
    axarr[1].set_xlabel(r"$M\ {\rm M_\odot}/h$")
    axarr[0].set_ylabel(r"number in bin")
    axarr[1].set_ylabel(r"residual realizations")
    axarr[1].set_ylim(-0.05, 0.05)
    plt.subplots_adjust(hspace=0.0, left=0.15, bottom=0.15)
    plt.show()
