"""
Plot the residuals generated.
"""
import numpy as np
import matplotlib.pyplot as plt
import aemulus_data as AD
import aemHMF
plt.rc("text", usetex=True)
plt.rc("font", size=18)
plt.rc("font", family="serif")

def peakheight_test(M, box=0):
    hmf = aemHMF.Aemulus_HMF()
    #nu has locations of the ticks in the x axis
    #Pick some cosmology, box=16
    Ombh2, Omch2, w, ns, ln10As, H0, Neff, sig8 = np.genfromtxt(AD.path_to_test_box_cosmologies())[int(box)]
    h = H0/100.
    Ob = Ombh2/h**2
    Oc = Omch2/h**2
    Om = Ob + Oc
    cosmo = {"om":Om, "ob":Ob, "ol":1-Om, "h":h, "s8":sig8, "ns":ns, "w0":w, "Neff":Neff, "wa":0}
    hmf.set_cosmology(cosmo)
    a = 0.25
    return np.array([aemHMF.peak_height(Mi, a) for Mi in M])

def get_masses(nu, box=0):
    hmf = aemHMF.Aemulus_HMF()
    #nu has locations of the ticks in the x axis
    #Pick some cosmology, box=16
    Ombh2, Omch2, w, ns, ln10As, H0, Neff, sig8 = np.genfromtxt(AD.path_to_test_box_cosmologies())[int(box)]
    h = H0/100.
    Ob = Ombh2/h**2
    Oc = Omch2/h**2
    Om = Ob + Oc
    cosmo = {"om":Om, "ob":Ob, "ol":1-Om, "h":h, "s8":sig8, "ns":ns, "w0":w, "Neff":Neff, "wa":0}
    hmf.set_cosmology(cosmo)
    Ms = np.logspace(11, 17, num=100) #Msun/h
    a = 1.0
    nus = np.array([aemHMF.peak_height(Mi, a) for Mi in Ms])
    from scipy.interpolate import interp1d
    spl = interp1d(nus, Ms)
    return spl(nu)#np.log10(spl(nu))
    

def get_resids(path, use_nu=True):
    return np.genfromtxt(path, unpack=True)


if __name__ == "__main__":
    use_nu = True
    zs, lMs, nus, R, Re, N, err, Nt08, box, snap = get_resids("BB_residuals.txt", use_nu)
    if use_nu: x = nus
    else: x = lMs
    plt.plot(x, R, marker='.', ls='', markersize=1, c='b', label="Building boxes", alpha=0.4)
    print "LOO residuals"
    w = 1./Re**2
    print np.mean(R), np.sum(w*R)/np.sum(w), np.sum(w*Re)/np.sum(w)
    print np.mean(Re), np.sqrt(np.mean(Re**2))



    zs, lMs, nus, R, Re, N, err, Nt08, box, snap = get_resids("test_residuals.txt", use_nu)
    if use_nu: x = nus
    else: x = lMs
    plt.plot(x, R, marker='.', ls='', markersize=2, c='r', label="Test boxes", alpha=1)
    w = 1./Re**2
    print "Testbox residuals"
    print np.mean(R), np.sum(w*R)/np.sum(w), np.sum(w*Re)/np.sum(w)
    print np.mean(Re), np.sqrt(np.mean(Re**2))

    """
    print "test"
    ind = 61
    print box[ind]
    print zs[ind:ind+3]
    print 10**lMs[ind:ind+3]
    print nus[ind:ind+3]
    print get_masses(nus[ind:ind+3], box=box[ind])
    """
    
    zs, lMs, nus, R, Re, N, err, Nt08, box, snap = get_resids("bestfit_residuals.txt", use_nu)
    Rvar = Re**2
    w = 1./Rvar
    if use_nu: x = nus
    else: x = lMs
    #plt.plot(x, R, marker='.', ls='', markersize=2, c='yellow', label="Best fits", alpha=1)

    
    plt.axhline(0, ls='-', c='k')
    inds = (box==0)*(snap==9)
    shot = 1./np.sqrt(N[inds])
    sample = err[inds]/N[inds]#* np.sqrt(5.)
    xi = x[inds]
    #plt.plot(xi, shot, ls='--', marker='', c='k', label=r"$1/\sqrt{N}$ at z=0")
    #plt.plot(xi, -shot, ls='--', marker='', c='k')
    #plt.plot(xi, sample, ls=':', marker='', c='k', label=r"$\sigma/N$ at z=0")
    #plt.plot(xi, -sample, ls=':', marker='', c='k')

    
    inds = (box==0)*(snap==2)
    shot = 1./np.sqrt(N[inds])
    sample = err[inds]/N[inds]
    xi = x[inds]
    """
    plt.plot(xi, shot, ls='--', marker='', c='gray', label=r"$1/\sqrt{N}$ at z=1")
    plt.plot(xi, -shot, ls='--', marker='', c='gray')
    plt.plot(xi, sample, ls=':', marker='', c='gray', label=r"$\sigma/N$ at z=1")
    plt.plot(xi, -sample, ls=':', marker='', c='gray')
    """

    
    ylim = .1
    plt.ylim(-ylim, ylim)
    plt.legend(loc=0, frameon=False, fontsize=14)
    if use_nu: plt.xlabel(r"$\nu$")
    else: plt.xlabel(r"$\log_{10}M$")
    #plt.ylabel(r"$\%$ Diff")
    plt.ylabel(r"$\frac{N-N_{emu}}{N_{emu}}$")

    plt.subplots_adjust(bottom=0.13, left=0.20, top=0.85)
    xlim = plt.gca().get_xlim()
    plt.fill_between(xlim, -0.01, 0.01, color='gray', alpha=0.2)
    plt.xlim(xlim)

    """
    if use_nu:
        #put mass on the top axis
        ax1 = plt.gca()
        ax2 = ax1.twiny()
        #masses = get_masses(ax1.get_xticks()[1])
        x2lim = 10**np.array([12.682920863623462, 15.900557388302612])#the mass limits ifwe were to use_nu=False
        ax2.set_xlim(x2lim)
        ax2.set_xlabel(r"Mass ${\rm M_\odot}\ h^{-1}$ at z=0")
        ax2.set_xscale("log")
        ax1.set_xlim(xlim)
        print "Mass xlims:", plt.gca().get_xlim()
        Masses = np.array([1e13, 1e14, 1e15])
        print "PH test"
        print Masses
        print peakheight_test(Masses)
    """

    
    plt.gcf().savefig("all_residuals.png")
    plt.show()
