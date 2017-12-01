import aemHMF
import numpy as np
import matplotlib.pyplot as plt
import aemulus_data as AD
plt.rc("text", usetex=True)
plt.rc("font", size=18, family='serif')
import tinker_mass_function as TMF


def get_tinker_lines(cosmo, M, a, Mbins):
    z = 1./a - 1.
    tmf = TMF.tinker_mass_function(cosmo, z)
    lMbins = np.log10(Mbins)
    return tmf.n_in_bins(lMbins)
    
    
if __name__ == "__main__":
    Volume = 1050.**3 #Mpc^3/h^3
    box = 2
    Ombh2, Omch2, w, ns, ln10As, H0, Neff, sig8 = np.genfromtxt(AD.path_to_test_box_cosmologies())[box]
    h = H0/100.
    Ob = Ombh2/h**2
    Oc = Omch2/h**2
    Om = Ob + Oc
    
    cosmo = {"om":Om, "ob":Ob, "ol":1-Om, "h":h, "s8":sig8, "ns":ns, "w0":w, "Neff":Neff}


    hmf = aemHMF.Aemulus_HMF()
    hmf.set_cosmology(cosmo)
    
    f, axarr = plt.subplots(2, sharex=True)
    sfs = AD.get_scale_factors()
    zs = 1./sfs - 1
    colors = [plt.get_cmap("seismic")(ci) for ci in np.linspace(1.0, 0.0, len(sfs))]
    for snapshot in range(len(sfs)):
        if snapshot < 0: continue
        a = sfs[snapshot]
        z = zs[snapshot]
        path = AD.path_to_test_box_data(box, snapshot)
        lMlo, lMhi, N, Mtot = np.genfromtxt(path, unpack=True)
        M_bins = 10**np.array([lMlo, lMhi]).T
        M = Mtot/N
        covpath = AD.path_to_test_box_covariance(box, snapshot)
        cov = np.loadtxt(covpath)
        err = np.sqrt(np.diag(cov))

        N_aem = hmf.n_bins(M_bins, a)*Volume
        pdiff = (N-N_aem)/N_aem
        pdiff_err = err/N_aem

        if snapshot in [0, 2, 5, 9]:
            axarr[0].errorbar(M, N, err, ls='', marker='.', c=colors[snapshot], label=r"$z=%.2f$"%z)
        else:
            axarr[0].errorbar(M, N, err, ls='', marker='.', c=colors[snapshot])
        axarr[0].loglog(M, N_aem, ls='-', c=colors[snapshot])
        #axarr[1].errorbar(M, pdiff, pdiff_err, c=colors[snapshot], ls='', marker='.')
        axarr[1].plot(M, pdiff, c=colors[snapshot], ls='', marker='.')


        #Get the Tinker lines
        Nt = get_tinker_lines(cosmo, M, a, M_bins)*Volume
        pd = (N - Nt)/Nt
        axarr[1].plot(M, pd, c=colors[snapshot], ls='-', alpha=0.2)

        
    axarr[1].axhline(0, c='k', ls='--')
    axarr[0].set_yscale('log')
    axarr[0].set_ylim(1, 1e6)
    plim = 0.25
    axarr[1].set_ylim(-plim, plim)
    plt.subplots_adjust(hspace=0, wspace=0, left=0.18, bottom=0.15)
    axarr[0].legend(loc="upper right", frameon=0, fontsize=12)
    axarr[1].set_xlabel(r"Mass $[{\rm M_\odot} h^{-1}]$")
    axarr[1].set_ylabel(r"$\frac{N-N_{model}}{N_{model}}$")
    axarr[0].set_ylabel(r"Number per bin")
    axarr[0].set_ylabel(r"Number of halos")

    xlim = axarr[1].get_xlim()
    axarr[1].fill_between(xlim, -0.01, 0.01, color="gray", alpha=0.2)
    axarr[1].set_xlim(xlim)
    plt.gcf().savefig("tinker_figure.pdf")
    plt.show()
