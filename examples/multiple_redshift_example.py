import aemHMF
import numpy as np
import matplotlib.pyplot as plt
import aemulus_data as AD


if __name__ == "__main__":

    Volume = 1050.**3 #Mpc^3/h^3
    box = 2
    Ombh2, Omch2, w, ns, ln10As, H0, Neff, sig8 = np.genfromtxt(AD.path_to_building_box_cosmologies())[box]
    h = H0/100.
    Ob = Ombh2/h**2
    Oc = Omch2/h**2
    Om = Ob + Oc
    
    cosmo = {"om":Om, "ob":Ob, "ol":1-Om, "h":h, "s8":sig8, "ns":ns, "w0":w, "Neff":Neff}
    #Extra shit
    cosmo['wa']=0.0

    hmf = aemHMF.Aemulus_HMF()
    print "here"
    hmf.set_cosmology(cosmo)
    print "now here"
    hmf.n_t08.t08_slopes_intercepts=np.array([ 2.36432263 , 0.54244443,  0.44712662 , 0.2049388  , 1.26487542,  0.15319037])
    
    f, axarr = plt.subplots(2, sharex=True)
    sfs = AD.get_scale_factors()
    for i in range(len(sfs)):
        if i > 0: continue
        a = sfs[i]
        snap = i
        path = AD.path_to_building_box_data(box, snap)
        lMlo, lMhi, N, Mtot = np.genfromtxt(path, unpack=True)
        M_bins = 10**np.array([lMlo, lMhi]).T
        M = Mtot/N

        N_aem = hmf.n_bins(M_bins, a, with_f=False)*Volume
        print hmf.n_t08.t08_params, hmf.n_t08.B
        print hmf.n_t08.dndlM(1e14, a)
        print "sigma = ",hmf.n_t08.sigmaM_spline(1e14, a)
        pdiff = (N-N_aem)/N_aem
    
        axarr[0].loglog(M, N, ls='', marker='.', c='k')
        axarr[0].loglog(M, N_aem, ls='-', c='b')
        axarr[1].plot(M, pdiff, c='b', ls='-')
    axarr[1].axhline(0, c='k', ls='--')
    axarr[0].set_yscale('log')
    plt.show()
