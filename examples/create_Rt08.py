"""
Here we create the Rt08 residuals, or
(N_sim - N_t08)/N_t08 = Rt08.

To do this, we can create an aemHMF object that we call with the flag: with_f=False.
"""
import aemHMF
import numpy as np
#Got rid of the "helper_routines" dependence since I can now use the
#Aemulus_data package.
import aemulus_data as AD
import matplotlib.pyplot as plt

scale_factors = AD.get_scale_factors()
zs = 1./scale_factors - 1.
Volume = 1050.**3 #Mpc/h ^3

def get_residuals(box, snapshot, hmf):
    a = scale_factors[snapshot]
    
    path = AD.path_to_building_box_data(box, snapshot)
    lMlo, lMhi, N, Mtot = np.genfromtxt(path, unpack=True)
    M = Mtot/N
    Mbins = 10**np.array([lMlo, lMhi]).T

    covpath = AD.path_to_building_box_covariance(box, snapshot)
    cov = np.loadtxt(covpath)
    err = np.sqrt(np.diag(cov))

    nt08 = hmf.n_bins(Mbins, a, with_f=False)
    Nt08 = nt08*Volume
    Residual = (N-Nt08)/Nt08
    Residerr = err/Nt08

    #Make arrays of everything that we want to return
    z = np.ones_like(N)*zs[snapshot]
    nu = np.array([aemHMF.peak_height(Mi, a) for Mi in M])
    boxnum_arr  = np.ones_like(N)*box
    snapnum_arr = np.ones_like(N)*box
    #axarr[0].errorbar(M, N, err)
    #axarr[0].loglog(M, Nt08)
    #axarr[1].errorbar(M, Residual, Residerr) 
    
    return z, np.log10(M), nu, Residual, Residerr, boxnum_arr, snapnum_arr

if __name__ == "__main__":
    hmf = aemHMF.Aemulus_HMF()
    N_boxes = 40
    N_snaps = 10
    z_arr = np.array([])
    lM_arr = np.array([])
    nu_arr = np.array([])
    Residuals = np.array([])
    Resid_err = np.array([])
    boxnum_arr = np.array([])
    snapnum_arr = np.array([])
    #Loop over all boxes and snapshots
    #the residuals from a given snapshot will be various
    #lengths, because some snapshots have different numbers
    #of bins in them, just by chance.
    for i in range(N_boxes):
        Ombh2, Omch2, w, ns, ln10As, H0, Neff, sig8 = np.genfromtxt(AD.path_to_building_box_cosmologies())[i]
        h = H0/100.
        Ob = Ombh2/h**2
        Oc = Omch2/h**2
        Om = Ob + Oc
        cosmo = {"om":Om, "ob":Ob, "ol":1-Om, "h":h, "s8":sig8, "ns":ns, "w0":w, "Neff":Neff}
        hmf.set_cosmology(cosmo)
        #fig, axarr = plt.subplots(2, sharex=True)
        for j in range(N_snaps):
            z, lM, nu, R, eR, box, snap = get_residuals(i, j, hmf)
            z_arr = np.concatenate([z_arr, z])
            lM_arr = np.concatenate([lM_arr, lM])
            nu_arr = np.concatenate([nu_arr, nu])
            Residuals = np.concatenate([Residuals, R])
            Resid_err = np.concatenate([Resid_err, eR])
            boxnum_arr = np.concatenate([boxnum_arr, box])
            snapnum_arr = np.concatenate([snapnum_arr, snap])
            print "done with ",i,j
        #plt.show()
    out = np.array([z_arr, lM_arr, nu_arr, Residuals, Resid_err, boxnum_arr, snapnum_arr]).T
    np.savetxt("R_T08.txt", out)
    print "done"
            
    
    
