import aemHMF
import numpy as np
import matplotlib.pyplot as plt
import aemulus_data as AD
plt.rc("text", usetex=True)
plt.rc("font", size=14, family='serif')

if __name__ == "__main__":
    box, snap = 0, 9
    path = AD.path_to_test_box_data(box, snap)
    Ombh2, Omch2, w, ns, ln10As, H0, Neff, sig8 = np.genfromtxt(AD.path_to_test_box_cosmologies())[box]
    h = H0/100.
    Ob = Ombh2/h**2
    Oc = Omch2/h**2
    Om = Ob + Oc
    cosmo = {"om":Om, "ob":Ob, "ol":1-Om, "h":h, "s8":sig8, "ns":ns, "w0":w, "Neff":Neff}

    hmf = aemHMF.Aemulus_HMF()
    hmf.set_cosmology(cosmo)
    M = np.logspace(11, 15, num=30)
    for a in [1.0, 0.5, 0.33333333, 0.25]:
        z = 1./a - 1
        sigma = hmf.Mtosigma(M, a)
        G = hmf.multiplicity(M, a)
        deltac = 1.686
        nu = deltac/sigma
        plt.plot(M, G, label=r"$z=%.1f$"%z)
        print "z = %f done"%z
    plt.title(r"$M_{low}$=%.1e  $M_{high}$=%.1e"%(min(M), max(M)))
    plt.ylabel(r"$G(\sigma)$")
    plt.xlabel(r"$\nu$")
    plt.xlabel(r"Mass [M$_\odot\ h^{-1}$]")
    plt.xscale('log')
    plt.legend(loc=0, frameon=False)
    plt.show()
