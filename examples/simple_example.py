import aemHMF
import numpy as np
import matplotlib.pyplot as plt
import aemulus_data as AD


if __name__ == "__main__":

    z = 0.0
    box, snap = 0, 9
    lMlo, lMhi, N, Mtot = AD.get_building_box_binned_mass_function(box, snap).T
    M_bins = 10**np.array([lMlo, lMhi]).T
    M = Mtot/N
    Volume = 1050.**3 #Mpc^3/h^3

    Ombh2, Omch2, w, ns, ln10As, H0, Neff, sig8 = np.genfromtxt(AD.path_to_test_box_cosmologies())[box]
    h = H0/100.
    Ob = Ombh2/h**2
    Oc = Omch2/h**2
    Om = Ob + Oc
    
    cosmo = {"Omega_m":Om, "Omega_b":Ob, "h":h, "sigma8":sig8, "n_s":ns, "w0":w, "N_eff":Neff}

    hmf = aemHMF.Aemulus_HMF()
    hmf.set_cosmology(cosmo)

    N_aem = hmf.n_bins(M_bins, z)*Volume
    pdiff = (N-N_aem)/N_aem
    
    f, axarr = plt.subplots(2, sharex=True)
    axarr[0].loglog(M, N, ls='', marker='.', c='k')
    axarr[0].loglog(M, N_aem, ls='-', c='b')
    axarr[0].set_yscale('log')

    axarr[1].plot(M, pdiff, c='b', ls='-')
    axarr[1].axhline(0, c='k', ls='--')
    plt.show()
