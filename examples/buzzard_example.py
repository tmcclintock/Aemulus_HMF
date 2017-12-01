"""
Here we fit to the darksy data.
"""
import aemHMF
import numpy as np
import matplotlib.pyplot as plt
plt.rc("text", usetex=True)
plt.rc("font", size=18, family="serif")

dsVolume = 8000.**3 #(Mpc/h)^3
bolshoipVolume = 250.**3 #(Mpc/h)^3
buzzardVolume = 250.**3 #(Mpc/h)^3

Ol = 0.7048737821671822
Om = 0.295037918703847
h = 0.6880620000000001
Ob = 0.04676431995034128
sig8 = 0.8344
w0 = -1.0
ns = 0.9676
darksky_cosmo = {"om":Om, "ob":Ob, "ol":Ol, "h":h, "s8":sig8, "ns":ns, "w0":w0, "wa":0.0, "Neff":3.046}

Om = 0.295
Ol = 0.705
Ob = 0.048
sig8 = 0.823
h = 0.678
ns = 0.968
w = -1
bolshoip_cosmo = {"om":Om, "ob":Ob, "ol":Ol, "h":h, "s8":sig8, "ns":ns, "w0":w0, "wa":0.0, "Neff":3.046}

Om = 0.286
Ol = 0.714
Ob = 0.047
sig8 = 0.82
h = 0.7
ns = 0.96
w = -1
buzzard_cosmo = {"om":Om, "ob":Ob, "ol":Ol, "h":h, "s8":sig8, "ns":ns, "w0":w0, "wa":0.0, "Neff":3.046}

def get_buzzard_data(a): #Masses are Msun/h, volume is [Mpc^3/h^3] comoving
    lMlo, lMhi, N, Mave = np.loadtxt("othersims_hmfs/buzzard_MF_a%.3f.txt"%a, unpack=True)
    C = np.loadtxt("othersims_hmfs/buzzard_C_a%.3f.txt"%a)
    return lMlo, lMhi, N, Mave, C

if __name__ == "__main__":
    #0.257, 0.506, 1.000
    a = 1.0 #this is the scale factor
    z = 1./a - 1.
    lMlo, lMhi, N, Mave, C = get_buzzard_data(a)
    Mlo = 10**lMlo
    Mhi = 10** lMhi
    M_bins = np.array([Mlo, Mhi]).T
    edges = np.append(Mlo, Mhi[-1])
    err = np.sqrt(np.diag(C))

    hmf = aemHMF.Aemulus_HMF()
    hmf.set_cosmology(buzzard_cosmo)
    N_aem = hmf.n_bins(M_bins, a)*buzzardVolume
    Nreal = 30
    fs = hmf.residual_realization(Mave, a, Nreal)

    fig, axarr = plt.subplots(2, sharex=True)
    axarr[0].errorbar(Mave, N, err, ls='', marker='o', c='k', label="Buzzard")
    axarr[0].loglog(Mave, N_aem, ls='-', c='b', label=r"Aemulus")
    #axarr[0].loglog(M, dndlM_cc, ls='-', c='r', label="Tinker08")
    axarr[0].set_ylabel(r"Number in bin")
    axarr[0].legend(loc=0, frameon=False)

    pdiff = (N - N_aem)/N_aem
    pde = err/N_aem
    
    axarr[1].errorbar(Mave, pdiff, pde, c='b', ls='-', marker='.', markersize=.1)
    #axarr[1].plot(M, pdcc, c='r', ls='-')
    axarr[1].set_ylabel(r"$\frac{N-N_{aem}}{N_{aem}}$")
    axarr[1].set_xlabel(r"$M [{\rm M}_\odot h^{-1}]$")
    ylim = 0.11
    axarr[1].set_ylim(-ylim, ylim)
    axarr[0].set_title(r"z=%.2f"%z)
    axarr[1].axhline(0, c='k', ls='--')
    xlim = axarr[1].get_xlim()
    axarr[1].fill_between(xlim, -0.01, 0.01, color="gray", alpha=0.2)
    axarr[1].set_xlim(xlim)

    for i in range(Nreal):
        Ns = N_aem * (1+fs[i])
        #axarr[0].loglog(Mave, Ns, ls='-', c='r', alpha=0.2)
        axarr[1].plot(Mave, (N-Ns)/N_aem, c="b", alpha=0.2)
    
    plt.subplots_adjust(hspace=0, left=0.18, bottom=0.15)
    plt.show()
