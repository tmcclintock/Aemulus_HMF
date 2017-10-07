import aemHMF
import numpy as np
import matplotlib.pyplot as plt
import aemulus_data as AD


if __name__ == "__main__":

    a = 1.0
    box, snap = 0, 9
    path = AD.path_to_building_box_data(box, snap)
    lMlo, lMhi, N, Mtot = np.genfromtxt(path, unpack=True)
    M_bins = 10**np.array([lMlo, lMhi]).T
    M = Mtot/N
    Volume = 1050.**3 #Mpc^3/h^3

    Ombh2, Omch2, w, ns, ln10As, H0, Neff, sig8 = np.genfromtxt(AD.path_to_building_box_cosmologies())[box]
    h = H0/100.
    Ob = Ombh2/h**2
    Oc = Omch2/h**2
    Om = Ob + Oc
    
    cosmo = {"om":Om, "ob":Ob, "ol":1-Om, "h":h, "s8":sig8, "ns":ns, "w0":w, "Neff":Neff}
    print cosmo
    hmf = aemHMF.Aemulus_HMF()
    hmf.set_cosmology(cosmo)
    N_aem = hmf.n_in_bins(M_bins, a, with_f=False)*Volume
    
    
    f, axarr = plt.subplots(2, sharex=True)
    axarr[0].loglog(M, N, ls='', marker='.', c='k')
    axarr[0].loglog(M, N_aem, ls='-', c='b')
    axarr[0].set_yscale('log')

    #axarr[1].plot(M, pdiff, c='b', ls='-')
    axarr[1].axhline(0, c='k', ls='--')
    plt.show()
