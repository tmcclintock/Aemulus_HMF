import aemHMF
import numpy as np
import matplotlib.pyplot as plt
import aemulus_data as AD
plt.rc("text", usetex=True)
plt.rc("font", size=18, family='serif')

if __name__ == "__main__":
    Volume = 1050.**3 #Mpc^3/h^3
    box = 0
    Ombh2, Omch2, w, ns, ln10As, H0, Neff, sig8 = AD.get_test_box_cosmologies()[box]
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
        if snapshot in [0, 1, 3, 4, 5, 7, 8]: continue
        a = sfs[snapshot]
        z = zs[snapshot]
        #path = AD.path_to_test_box_data(box, snapshot)
        #data = np.genfromtxt(path)
        data = AD.get_test_box_binned_mass_function(box, snapshot)
        N = data[:, 2]
        inds = (N > 0)
        data = data[inds]
        lMlo, lMhi, N, Mtot  = data.T
        M_bins = 10**np.array([lMlo, lMhi]).T
        M = Mtot/N
        #covpath = AD.path_to_test_box_covariance(box, snapshot)
        #cov = np.loadtxt(covpath)
        cov = AD.get_test_box_binned_mass_function_covariance(box, snapshot)
        err = np.sqrt(np.diag(cov))[inds]

        N_aem = hmf.n_bins(M_bins, a)*Volume
        pdiff = (N-N_aem)/N_aem
        pdiff_err = err/N_aem

        Mf = np.logspace(np.log10(min(M)), np.log10(max(M)), num=100)
        fs = hmf.residual_realization(M, a, 30)
        for i in range(len(fs)):
            Ns = N_aem*(1+fs[i])
            axarr[1].plot(M, (N-Ns)/N_aem, c=colors[snapshot], alpha=0.2)
            
        axarr[0].errorbar(M, N, err, ls='', marker='.', c=colors[snapshot], label=r"$z=%.2f$"%z)
        axarr[0].loglog(M, N_aem, ls='-', c=colors[snapshot])
        axarr[1].errorbar(M, pdiff, pdiff_err, c=colors[snapshot], ls='', marker='.')
    axarr[1].axhline(0, c='k', ls='--')
    axarr[0].set_yscale('log')
    axarr[0].set_ylim(1, 1e6)
    plim = 0.07
    axarr[1].set_ylim(-plim, plim)
    plt.subplots_adjust(hspace=0, wspace=0, left=0.2, bottom=0.15)
    axarr[0].legend(loc=0, frameon=0, fontsize=14)
    axarr[1].set_xlabel(r"Mass $[{\rm M_\odot}\ h^{-1}]$")
    #axarr[1].set_ylabel(r"$\Delta N/N_{emu}$")
    axarr[1].set_ylabel(r"$\frac{N-N_{emu}(1+R_{GP})}{N_{emu}}$")
    axarr[0].set_ylabel(r"Number per bin")
    xlim = axarr[1].get_xlim()
    axarr[1].fill_between(xlim, -0.01, 0.01, color="gray", alpha=0.2)
    axarr[1].set_xlim(xlim)
    plt.gcf().savefig("scatter_figure.pdf")
    plt.gcf().savefig("scatter_figure.png")
    plt.show()
