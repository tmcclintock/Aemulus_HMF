import aemHMF
import numpy as np
import matplotlib.pyplot as plt
import aemulus_data as AD
plt.rc("text", usetex=True)
plt.rc("font", size=18)
plt.rc('font', family='serif')

Rmatrix = np.loadtxt("../aemHMF/data_files/R_matrix.txt")
bfparams_all = np.loadtxt("../aemHMF/data_files/rotated_dfg_means.txt")

if __name__ == "__main__":
    Volume = 1050.**3 #Mpc^3/h^3
    box = 1
    Ombh2, Omch2, w, ns, ln10As, H0, Neff, sig8 = AD.building_box_cosmologies()[box]
    cosmo={'Obh2':Ombh2, 'Och2':Omch2, 'w0':w, 'n_s':ns, 'ln10^{10}A_s':ln10As, 'N_eff':Neff, 'H0':H0}
    
    bfparams = bfparams_all[box]
    
    hmf = aemHMF.Aemulus_HMF()
    hmf.set_cosmology(cosmo)
    #hmf.n_t08.t08_slopes_intercepts = np.dot(Rmatrix, bfparams).flatten()
    
    f, axarr = plt.subplots(2, sharex=True)
    sfs = AD.scale_factors()
    zs = 1./sfs - 1
    colors = [plt.get_cmap("seismic")(ci) for ci in np.linspace(1.0, 0.0, len(sfs))]
    for snapshot in range(len(sfs)):
        if snapshot < 0: continue
        a = sfs[snapshot]
        z = zs[snapshot]
        lMlo, lMhi, N, Mtot = AD.building_box_binned_mass_function(box, snapshot).T
        M_bins = 10**np.array([lMlo, lMhi]).T
        M = Mtot/N
        cov = AD.building_box_binned_mass_function_covariance(box, snapshot)
        err = np.sqrt(cov.diagonal())

        N_aem = hmf.n_bins(M_bins, z)*Volume
        pdiff = (N-N_aem)/N_aem
        pdiff_err = err/N_aem

        icov = np.linalg.inv(cov)
        X = N - N_aem
        chi2 = np.dot(X, np.dot(icov, X))
        print "chi2(%d) = %.2f"%(snapshot, chi2)

        if snapshot in [0, 2, 5, 9]:
            axarr[0].errorbar(M, N, err, ls='', marker='.', c=colors[snapshot], label=r"$z=%.2f$"%z)
        else:
            axarr[0].errorbar(M, N, err, ls='', marker='.', c=colors[snapshot])

        axarr[0].loglog(M, N_aem, ls='-', c=colors[snapshot])
        axarr[1].errorbar(M, pdiff, pdiff_err, c=colors[snapshot], ls='', marker='.')#, markersize=1)
    axarr[1].axhline(0, c='k', ls='--')
    axarr[0].set_yscale('log')
    axarr[0].set_ylim(1, 1e6)
    plim = 0.07
    axarr[1].set_ylim(-plim, plim)
    axarr[0].legend(loc=0, frameon=0, fontsize=8)
    axarr[1].set_xlabel(r"Mass $[{\rm M_\odot} h^{-1}]$")
    axarr[1].set_ylabel(r"$\frac{N-N_{fit}}{N_{fit}}$")
    axarr[0].set_ylabel(r"Number per bin")
    xlim = axarr[1].get_xlim()
    axarr[1].fill_between(xlim, -0.01, 0.01, color="gray", alpha=0.2)
    axarr[1].set_xlim(xlim)
    plt.subplots_adjust(hspace=0, wspace=0, left=0.20, bottom=0.15)
    plt.gcf().savefig("bestfit_final.pdf")
    plt.show()
