"""
Here we create the Rt08 residuals, or
(N_sim - N_t08)/N_t08 = Rt08.

To do this, we can create an aemHMF object that we call with the flag: with_f=False.
"""
import aemHMF
import numpy as np
import aemulus_data as AD
import matplotlib.pyplot as plt

scale_factors = AD.scale_factors()
zs = 1./scale_factors - 1.
Volume = 1050.**3 #Mpc/h ^3

model = "defg"
with_f = False
#Rmatrix = np.loadtxt("../aemHMF/data_files/R_matrix.txt")

def get_residuals(box, snapshot, hmf, building_boxes=True, bf=False):
    if building_boxes:
        get_data = AD.building_box_binned_mass_function
        get_cov = AD.building_box_binned_mass_function_covariance
    else:
        get_data = AD.test_box_binned_mass_function
        get_cov = AD.test_box_binned_mass_function_covariance
    a = scale_factors[snapshot]
    z = zs[snapshot]
    lMlo, lMhi, N, Mtot = get_data(box, snapshot)
    cov = get_cov(box, snapshot)
    good = N > 0
    lMlo = lMlo[good]
    lMhi = lMhi[good]
    N = N[good]
    Mtot = Mtot[good]
    cov = cov[good]
    cov = cov[:,good]
    icov = np.linalg.inv(cov)
    err = np.sqrt(np.diag(cov))
    M = Mtot/N
    Mbins = 10**np.array([lMlo, lMhi]).T

    if bf:
        bfparams = np.loadtxt("../aemHMF/data_files/r_%s_means.txt"%model)[box]
        hmf.tinkerMF.t08_slopes_intercepts = np.dot(Rmatrix, bfparams).flatten()
        hmf.tinkerMF.merge_t08_params(a)
    
    n_aem = hmf.n_bins(Mbins, z)
    N_aem = nt08*Volume
    Residual = (N-N_aem)/N_aem
    Residerr = err/N_aem

    #Make arrays of everything that we want to return
    z = np.ones_like(N)*zs[snapshot]
    nu = np.array([aemHMF.peak_height(Mi, a) for Mi in M])
    boxnum_arr  = np.ones_like(N)*box
    snapnum_arr = np.ones_like(N)*snapshot

    return z, np.log10(M), nu, Residual, Residerr, N, err, Nt08, boxnum_arr, snapnum_arr

def get_all_residuals(building_box=True, bf=False):
    if building_box:
        N_boxes = 40
        cospath = AD.path_to_building_box_cosmologies()
        outpath = "BB_residuals.txt"
        if with_f: outpath = "BB_residuals_f.txt"
    else:
        N_boxes = 7
        cospath = AD.path_to_test_box_cosmologies()
        outpath = "test_residuals.txt"
        if with_f: outpath = "test_residuals_f.txt"
    if bf:
        print "Working with best fits"
        N_boxes = 40
        cospath =  AD.path_to_building_box_cosmologies()
        outpath = "bestfit_residuals.txt"
    N_snaps = 10
    hmf = aemHMF.Aemulus_HMF()
    z_arr = np.array([])
    lM_arr = np.array([])
    nu_arr = np.array([])
    Residuals = np.array([])
    Resid_err = np.array([])
    N_arr = np.array([])
    err_arr = np.array([])
    Nt08_arr = np.array([])
    boxnum_arr = np.array([])
    snapnum_arr = np.array([])
    for i in range(N_boxes):
        Ombh2, Omch2, w, ns, ln10As, H0, Neff, sig8 = np.genfromtxt(cospath)[i]
        h = H0/100.
        Ob = Ombh2/h**2
        Oc = Omch2/h**2
        Om = Ob + Oc
        cosmo = {"om":Om, "ob":Ob, "ol":1-Om, "h":h, "s8":sig8, "ns":ns, "w0":w, "Neff":Neff}
        hmf.set_cosmology(cosmo)
        for j in range(N_snaps):
            z, lM, nu, R, eR, N, err, Nt08, box, snap = get_residuals(i, j, hmf, building_box, bf)
            z_arr = np.concatenate([z_arr, z])
            lM_arr = np.concatenate([lM_arr, lM])
            nu_arr = np.concatenate([nu_arr, nu])
            Residuals = np.concatenate([Residuals, R])
            Resid_err = np.concatenate([Resid_err, eR])
            N_arr = np.concatenate([N_arr, N])
            err_arr = np.concatenate([err_arr, err])
            Nt08_arr = np.concatenate([Nt08_arr, Nt08])
            boxnum_arr = np.concatenate([boxnum_arr, box])
            snapnum_arr = np.concatenate([snapnum_arr, snap])
            print "Residuals made for the box at ",i,j
    out = np.array([z_arr, lM_arr, nu_arr, Residuals, Resid_err, N_arr, err_arr, Nt08_arr, boxnum_arr, snapnum_arr]).T
    np.savetxt(outpath, out)
    print "done, with_f = ",with_f, " building_box = ",building_box, " bestfits = ",bf
    
    
if __name__ == "__main__":
    #get_all_residuals(building_box=True)
    #get_all_residuals(building_box=False)
    get_all_residuals(building_box=True, bf=True)
