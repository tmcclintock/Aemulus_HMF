import aemHMF
import numpy as np
import matplotlib.pyplot as plt
import aemulus_data as AD
plt.rc("text", usetex=True)
plt.rc("font", size=18)
plt.rc('font', family='serif')

if __name__ == "__main__":
    Volume = 1050.**3 #Mpc^3/h^3
    box = 4
    Ombh2, Omch2, w, ns, ln10As, H0, Neff, sig8 = AD.test_box_cosmologies()[box]

    cosmo={'Obh2':Ombh2, 'Och2':Omch2, 'w0':w, 'n_s':ns, 'ln10^{10}A_s':ln10As, 'N_eff':Neff, 'H0':H0}#, 'sigma8':sig8}

    hmf = aemHMF.Aemulus_HMF()
    hmf.set_cosmology(cosmo)
    
    f, axarr = plt.subplots(2, sharex=True)
    sfs = AD.scale_factors()
    zs = 1./sfs - 1
    colors = [plt.get_cmap("seismic")(ci) for ci in np.linspace(1.0, 0.0, len(sfs))]
    for snapshot in range(len(sfs)):
        z = zs[snapshot]
        lMlo, lMhi, N, Mtot = AD.test_box_binned_mass_function(box, snapshot).T
        M_bins = 10**np.array([lMlo, lMhi]).T
        inds = (N>0)
        Mtot = Mtot[inds]
        N = N[inds]
        M = Mtot/N
        M_bins = M_bins[inds]
        cov = AD.test_box_binned_mass_function_covariance(box, snapshot)
        cov = cov[inds]
        cov = cov[:,inds]
        icov = np.linalg.inv(cov)
        err = np.sqrt(cov.diagonal())
        
        N_aem = hmf.n_bins(M_bins, z)*Volume
        pdiff = (N-N_aem)/N_aem
        pdiff_err = err/N_aem
        chi2 = np.dot(N-N_aem, np.dot(icov, N-N_aem))
        print "b%d s%d %.2f"%(box, snapshot, chi2)

        if snapshot in [0, 2, 5, 9]:
            axarr[0].errorbar(M, N, err, ls='', marker='.', c=colors[snapshot], label=r"$z=%.2f$"%z)
        else:
            axarr[0].errorbar(M, N, err, ls='', marker='.', c=colors[snapshot])
        axarr[0].loglog(M, N_aem, ls='-', c=colors[snapshot])
        axarr[1].errorbar(M, pdiff, pdiff_err, c=colors[snapshot], ls='', marker='.')
    axarr[1].axhline(0, c='k', ls='--')
    axarr[0].set_yscale('log')
    axarr[0].set_ylim(1, 1e6)
    plim = 0.07
    axarr[1].set_ylim(-plim, plim)
    axarr[0].legend(loc="upper right", frameon=0, fontsize=12)
    axarr[1].set_xlabel(r"Mass $[{\rm M_\odot} h^{-1}]$")
    axarr[1].set_ylabel(r"$\frac{N-N_{emu}}{N_{emu}}$")
    axarr[0].set_ylabel(r"Number per bin")
    xlim = axarr[1].get_xlim()
    axarr[1].fill_between(xlim, -0.01, 0.01, color="gray", alpha=0.2)
    axarr[1].set_xlim(xlim)
    plt.subplots_adjust(hspace=0, wspace=0, left=0.2, bottom=0.15)
    #plt.gcf().savefig("testbox_figure.pdf")
    plt.show()
