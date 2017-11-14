import aemHMF
import numpy as np
import matplotlib.pyplot as plt
import aemulus_data as AD
plt.rc("text", usetex=True)
plt.rc("font", size=18, family='serif')
import sys

deltac = 1.686

box, snap = 0, 9
path = AD.path_to_test_box_data(box, snap)
Ombh2, Omch2, w, ns, ln10As, H0, Neff, sig8 = np.genfromtxt(AD.path_to_test_box_cosmologies())[box]
h = H0/100.
Ob = Ombh2/h**2
Oc = Omch2/h**2
Om = Ob + Oc
default_cosmo = {"om":Om, "ob":Ob, "ol":1-Om, "h":h, "s8":sig8, "ns":ns, "w0":w, "Neff":Neff} #Default cosmology

def function_of_z():
    Ombh2, Omch2, w, ns, ln10As, H0, Neff, sig8 = np.genfromtxt(AD.path_to_test_box_cosmologies())[box]
    cosmo = default_cosmo.copy()
    hmf = aemHMF.Aemulus_HMF()
    hmf.set_cosmology(cosmo)
    
    sigma = np.linspace(1, 6, num=30)
    Gz0 = hmf.multiplicity_sigma(sigma, 1.0)
    for a in [0.5, 0.33333333, 0.25]:
        z = 1./a - 1
        G = hmf.multiplicity_sigma(sigma, a)
        nu = deltac/sigma
        Y = (G-Gz0)/Gz0
        plt.plot(M, Y, label=r"$z=%.1f$"%z)
        print "z = %f done"%z
    plt.title(r"$M_{low}$=%.1e  $M_{high}$=%.1e"%(min(M), max(M)))
    plt.ylabel(r"$\frac{G(z) - G(z=0)}{G(z=0)}$")
    plt.xlabel(r"$\nu$")
    plt.xlabel(r"Mass [M$_\odot\ h^{-1}$]")
    plt.xscale('log')
    plt.legend(loc="upper left", frameon=False)
    plt.show()

def function_of_Omegam(ax):
    hmf = aemHMF.Aemulus_HMF()
    a = 0.5
    z = 1./a - 1
    cosmo = default_cosmo.copy()
    Om = 0.3
    cosmo['om'] = Om
    cosmo['ol'] = 1. - Om
    hmf.set_cosmology(cosmo)
    nu = np.linspace(1, 6, num=30)
    sigma = deltac/nu
    Gom0 = hmf.multiplicity_sigma(sigma, a)

    for Om in [0.26, 0.28, 0.3, 0.32, 0.34]:
        cosmo['om'] = Om
        cosmo['ol'] = 1. - Om
        hmf.set_cosmology(cosmo)
        G = hmf.multiplicity_sigma(sigma, a)
        Y = (G - Gom0)/Gom0
        ax.plot(nu, Y, label=r"$\Omega_m=%.2f$"%Om)
        print "z = %f done"%z
    ax.set_ylabel(r"$\frac{G(\sigma) - G(\sigma)_{fid}}{G(\sigma)_{fid}}$")
    ax.set_xlabel(r"$\nu$")
    ax.legend(loc=0, frameon=False, fontsize=12)
    
def function_of_sigma8(ax):
    hmf = aemHMF.Aemulus_HMF()
    a = 0.5
    z = 1./a - 1
    cosmo = default_cosmo.copy()
    cosmo['s8'] = 0.8
    hmf.set_cosmology(cosmo)
    nu = np.linspace(1, 6, num=30)
    sigma = deltac/nu
    Gom0 = hmf.multiplicity_sigma(sigma, a)

    for sig8 in [0.7, .75, 0.8, 0.85, .9]:
        cosmo["s8"] = sig8
        hmf.set_cosmology(cosmo)
        G = hmf.multiplicity_sigma(sigma, a)
        Y = (G - Gom0)/Gom0
        ax.plot(nu, Y, label=r"$\sigma_8=%.2f$"%sig8)
        print "z = %f done"%z
    #ax.set_ylabel(r"$\frac{G(\sigma)-G(\sigma)_{fid}}{G(\sigma)_{fid}}$")
    ax.set_xlabel(r"$\nu$")
    ax.legend(loc="upper left", frameon=False, fontsize=12)

if __name__ == "__main__":
    fig, axes = plt.subplots(1, 2, sharey=True)
    function_of_Omegam(axes[0])
    function_of_sigma8(axes[1])
    plt.subplots_adjust(wspace=0, bottom=0.15, left=0.15)
    fig.set_size_inches(8, 4)
    #fig.savefig("multiplicity_figure.png")
    #fig.savefig("multiplicity_figure.pdf")
    plt.show()
