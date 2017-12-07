"""
Plot the residuals generated.
"""
import numpy as np
import matplotlib.pyplot as plt
import aemulus_data as AD
import aemHMF
plt.rc("text", usetex=True)
plt.rc("font", size=18, family='serif')
from matplotlib.ticker import MaxNLocator
import cosmocalc as cc
DELTAC=1.686

def get_masses(nu, z, box=0):
    Ombh2, Omch2, w, ns, ln10As, H0, Neff, sig8 = np.genfromtxt(AD.path_to_building_box_cosmologies())[int(box)]
    h = H0/100.
    Ob = Ombh2/h**2
    Oc = Omch2/h**2
    Om = Ob + Oc
    cosmo = {"om":Om, "ob":Ob, "ol":1-Om, "h":h, "s8":sig8, "ns":ns, "w0":w, "Neff":Neff, "wa":0}
    cc.set_cosmology(cosmo)
    Ms = np.logspace(8, 17, num=100) #Msun/h
    a = 1.0/(1+z)
    nus = np.array([DELTAC/cc.sigmaMtophat(Mi, a) for Mi in Ms])
    from scipy.interpolate import interp1d
    spl = interp1d(nus, np.log(Ms))
    return np.exp(spl(nu))#np.log10(spl(nu))

def get_colors(cmapstring="seismic"):
    cmap = plt.get_cmap(cmapstring)
    return [cmap(ci) for ci in np.linspace(1.0, 0.0, 10)]

scale_factors = np.array([0.25, 0.333333, 0.5, 0.540541, 0.588235, 
                          0.645161, 0.714286, 0.8, 0.909091, 1.0])
z = 1./scale_factors - 1.0


if __name__ == "__main__":


    colors = get_colors()
    
    use_nu = True
    zs, lMs, nus, R, Re, N, err, Nt08, box, snap =  np.genfromtxt("BB_residuals.txt", unpack=True)
    if use_nu: x = nus
    else: x = lMs
    
    fgp = aemHMF.Aemulus_HMF().residualgp
    nugp = np.linspace(min(x)-.1, max(x)+.1, 100)
    fig, axarr = plt.subplots(1,2, sharey=True)
    #First do the points
    ax = 0
    for i in range(len(z)):
        if i not in [1, 9]: continue
        inds = (z[i] == zs)
        #print lMs[inds]
        #print np.log10(get_masses(nus[inds], z[i]))
        axarr[ax].plot(x[inds], R[inds], marker='.', ls='', markersize=1, c=colors[i], label=r"$z=%.2f$"%z[i], alpha=0.4)
        zgp = z[i]*np.ones_like(nugp)
        mu, cov = fgp.predict_residual(nugp, zgp)
        errgp = np.sqrt(np.diag(cov))
        #axarr[ax].fill_between(nugp, -errgp, errgp, color=colors[i], alpha=0.1,zorder=-(i-10))
        for j in range(10):
            axarr[ax].plot(nugp, fgp.residual_realization(nugp, zgp), c=colors[i], ls='-', zorder=-(i-10), alpha=0.2)

        axarr[ax].axhline(0, ls='-', c='k', lw=1)
        #axarr[ax].fill_between([min(nugp), max(nugp)], -0.01, 0.01, color='gray', alpha=0.2)
        axarr[ax].set_ylim(-.1, .1)
        axarr[ax].set_xlabel(r"$\nu$")
        if ax==0: axarr[ax].set_ylabel(r"$\frac{N-N_{emu}}{N_{emu}}$")
        axarr[ax].set_xlim([min(nugp), max(nugp)])
        axarr[ax].set_xticks([1,2,3,4,5])
        axarr[ax].set_xticklabels([1,2,3,4,5])
        axarr[ax].set_title(r"$z=%.2f$"%z[i])
        #axarr[ax].legend(loc="upper left", frameon=False, fontsize=14)
        ax += 1

    plt.subplots_adjust(hspace=0, wspace=0, bottom=0.2, left=0.20)#, top=0.85)
    fig.set_size_inches(6, 3)
    #fig.savefig("residual_emulator_figure.png")
    #fig.savefig("residual_emulator_figure.pdf")
    plt.show()
