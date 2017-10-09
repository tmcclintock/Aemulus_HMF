import aemHMF
import numpy as np
import matplotlib.pyplot as plt
import aemulus_data as AD
import tinker_mass_function as TMF_mod

if __name__ == "__main__":

    Volume = 1050.**3 #Mpc^3/h^3
    box = 0
    Ombh2, Omch2, w, ns, ln10As, H0, Neff, sig8 = np.genfromtxt(AD.path_to_test_box_cosmologies())[box]
    h = H0/100.
    Ob = Ombh2/h**2
    Oc = Omch2/h**2
    Om = Ob + Oc
    
    cosmo = {"om":Om, "ob":Ob, "ol":1-Om, "h":h, "s8":sig8, "ns":ns, "w0":w, "Neff":Neff}
    sfs = AD.get_scale_factors()

    hmf = aemHMF.Aemulus_HMF()
    hmf.set_cosmology(cosmo)

    fig, axarr = plt.subplots(2, sharex=True)
    for i in range(len(sfs)):
        a = sfs[i]
        redshift = 1./a - 1

        hmf.n_t08.merge_t08_params(a)
        d,e,f,g = hmf.n_t08.t08_params

        #Create a TMF object
        TMF = TMF_mod.tinker_mass_function(cosmo,redshift=redshift)
        TMF.set_parameters(d,e,f,g)
        Mlo = 1e13
        Mhi = 1e14
        Mbin = np.array([[Mlo, Mhi]])
        l10Mbin = np.log10(Mbin)
        print TMF.n_in_bins(l10Mbin), hmf.n_t08.n_bin(Mlo, Mhi, a)

        M = np.logspace(11, 15, num=100)
        lM = np.log(M)
        dndlM_aem = np.array([hmf.n_t08.dndlM(Mi, a) for Mi in M])
        dndlM_tmf = np.array([TMF.dndlM(lMi) for lMi in lM])

        pd = (dndlM_aem - dndlM_tmf)/dndlM_tmf


        axarr[0].loglog(M, dndlM_aem, c='r')
        axarr[0].loglog(M, dndlM_tmf, ls='--', c='b')
        axarr[1].plot(M, pd, c='k')
    plt.show()
