"""
Here we fit to the darksy data.
"""
import aemHMF
import numpy as np
import matplotlib.pyplot as plt

#Scale factors that darksky is at
scale_factors = np.array([0.3333, 0.5, 0.6667, 0.8, 1.0])

def get_darksky_data(a): #Masses are Msun/h, volume is [Mpc^3/h^3] comoving
    path = "darksky_hmfs/ds14_a_halos_%.4f.hist8_m200b"%a
    bin_center_mass, dndlM, sigma, dlogsdlogm, lower_pmass, n, expected, dm, ds, dlnm, dlns = np.genfromtxt(path, skip_header=12, unpack=True)
    return bin_center_mass, dndlM

def get_prediction(bin_center_mass, a, cosmo):
    hmf = aemHMF.Aemulus_HMF()
    hmf.set_cosmology(cosmo)
    return np.array([hmf.n_t08.dndlM(M, a) for M in bin_center_mass])

def get_cc_prediction(bin_center_mass, a, cosmo):
    import cosmocalc as cc
    cc.set_cosmology(cosmo)
    return np.array([cc.tinker2008_mass_function(M, a, 200)*M for M in bin_center_mass])

if __name__ == "__main__":
    a = scale_factors[-1]
    M, dndlM = get_darksky_data(a)
    Om = 0.295126
    Ob = 0.0468
    h = 0.688
    sig8 = 0.835
    cosmo = {"om":Om, "ob":Ob, "ol":1-Om, "h":h, "s8":sig8, "ns":0.9676, "w0":-1, "Neff":3.08}
    dndlM_aem = get_prediction(M, a, cosmo)
    pdiff = (dndlM - dndlM_aem)/dndlM_aem
    pdiff = np.log(dndlM/dndlM_aem)

    dndlM_cc = get_cc_prediction(M, a, cosmo)
    #pdcc = (dndlM - dndlM_cc)/dndlM_cc
    pdcc = np.log(dndlM/dndlM_cc)

    print (dndlM_aem/dndlM)[10:20]

    f, axarr = plt.subplots(2, sharex=True)
    axarr[0].loglog(M, dndlM, ls='', marker='.', c='k', label="darksky")
    axarr[0].loglog(M, dndlM_aem, ls='-', c='b', label="Aemulus")
    axarr[0].loglog(M, dndlM_cc, ls='-', c='r', label="Tinker08")
    axarr[0].set_ylabel(r"$dn/d\ln M$")
    axarr[0].legend(loc=0)

    plt.subplots_adjust(hspace=0)
    axarr[1].plot(M, pdiff, c='b', ls='-')
    axarr[1].plot(M, pdcc, c='r', ls='-')
    axarr[1].set_xlabel(r"$M$")

    axarr[1].axhline(0, c='k', ls='--')
    plt.show()