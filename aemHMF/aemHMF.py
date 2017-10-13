"""
This contains the Aemulus HMF emulator.
"""
import os, sys
import n_t08
import f_gp
from n_t08 import peak_height
import numpy as np

#A placeholder cosmo_dict
cd = {"om":0.3,"ob":0.05,"ol":1.-0.3,"ok":0.0,"h":0.7,"s8":0.77,"ns":0.96,"w0":-1.0,"wa":0.0,"Neff":3.0}


class Aemulus_HMF(object):
    
    def __init__(self):
        self.n_t08 = None#n_t08.n_t08(cd)
        self.f = f_gp.f_gp()

    def set_cosmology(self, cosmo_dict):
        #This is a requirement for now
        cosmo_dict["wa"] = 0.0
        self.n_t08 = n_t08.n_t08()
        self.n_t08.set_cosmology(cosmo_dict)

    def n_bin(self, Mlow, Mhigh, a, with_f=True):
        n_t08 = self.n_t08.n_bin(Mlow, Mhigh, a)
        if not with_f: return n_t08
        M = np.mean(Mlow, Migh)
        nu = peak_height(M, a)
        z = 1-1./a
        f, fcov = self.f.predict_f(nu, np.ones_like(nu)*z)
        return n_t08*(1+f)

    def n_bins(self, Mbins, a, with_f=True, with_scatter=False):
        n_t08 = self.n_t08.n_bins(Mbins, a)
        if not with_f: return n_t08
        M = np.mean(Mbins, 1)
        nu = np.array([peak_height(Mi, a) for Mi in M])
        z = 1-1./a
        if not with_scatter: f, cov = self.f.predict_f(nu, np.ones_like(nu)*z)
        else: f = self.f.sample_f(nu, np.ones_like(nu)*z)
        return n_t08*(1.+f)

    def f_scatter(self, Mbins, a, N_samples):
        z = 1-1./a
        M = np.mean(Mbins, 1)
        nu = np.array([peak_height(Mi, a) for Mi in M])
        return np.array([self.f.sample_f(nu, np.ones_like(nu)*z) for i in range(N_samples)])

if __name__ == "__main__":
    cd = {"om":0.3,"ob":0.05,"ol":1.-0.3,"ok":0.0,"h":0.7,"s8":0.77,"ns":0.96,"w0":-1.0,"wa":0.0,"Neff":3.0}
    a = 1.0 #Scale factor
    hmf = Aemulus_HMF()
    hmf.set_cosmology(cd)
    M = np.logspace(11, 16, num=11, base=10)
    Mbins = np.array([M[:-1], M[1:]]).T
    Mave = np.mean(Mbins, 1)
    n = hmf.n_bins(Mbins, a)
    nt08 = hmf.n_bins(Mbins, a, with_f=False)
    V = 1050.**3 #Mpc/h ^3

    import matplotlib.pyplot as plt
    fig, axarr = plt.subplots(2, sharex=True)
    axarr[0].loglog(Mave, n*V, c='k', label=r"$N_{\rm emu}$")
    axarr[0].loglog(Mave, nt08*V, c='r', ls='--', label=r"$N_{\rm T08}$")
    plt.sca(axarr[0])
    plt.legend(loc=0, fontsize=14)
    axarr[1].scatter(Mave, (n-nt08)/nt08, c='k')
    fs = hmf.f_scatter(Mbins, a, 50)
    for i in range(len(fs)):
        ns = nt08*(1+fs[i])
        axarr[1].plot(Mave, (ns-nt08)/nt08, c='blue', alpha=0.2, zorder=-1)
    axarr[1].axhline(0, c='k', ls='-')
    axarr[1].set_xlabel(r"$M\ {\rm M_\odot}/h$")
    axarr[0].set_ylabel(r"number")
    axarr[1].set_ylabel(r"frac diff")
    axarr[1].set_ylim(-0.05, 0.05)
    plt.subplots_adjust(hspace=0.0, left=0.15, bottom=0.15)
    plt.show()
