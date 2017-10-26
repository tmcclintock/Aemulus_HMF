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

if __name__ == "__main__":
    fig, axes = plt.subplots(1, 2, sharey= True)
    
    zs, lMs, nus, R, Re, N, err, Nt08, box, snap = np.genfromtxt("BB_residuals.txt", unpack=True)
    M = 10**lMs
    axes[0].plot(nus, R, marker='.', ls='', markersize=1, c='b', label="Building boxes", alpha=0.4)
    axes[1].plot(M, R, marker='.', ls='', markersize=1, c='b', alpha=0.4)


    zs, lMs, nus, R, Re, N, err, Nt08, box, snap = np.genfromtxt("test_residuals.txt", unpack=True)
    M = 10**lMs
    axes[0].plot(nus, R, marker='.', ls='', markersize=1, c='r', label="Test boxes")
    axes[1].plot(M, R, marker='.', ls='', markersize=1, c='r')

    zs, lMs, nus, R, Re, N, err, Nt08, box, snap = np.genfromtxt("bestfit_residuals.txt", unpack=True)
    Rvar = Re**2
    w = 1./Rvar
    print np.mean(R), np.sum(w*R)/np.sum(w)
    print np.mean(Re)
    #plt.plot(x, R, marker='.', ls='', markersize=2, c='yellow', label="Best fits", alpha=1)

    
    axes[0].axhline(0, ls='-', c='k')
    axes[1].axhline(0, ls='-', c='k')
    inds = (box==0)*(snap==9)
    shot = 1./np.sqrt(N[inds])
    sample = err[inds]/N[inds]#* np.sqrt(5.)
    #xi = x[inds]
    #plt.plot(xi, shot, ls='--', marker='', c='k', label=r"$1/\sqrt{N}$ at z=0")
    #plt.plot(xi, -shot, ls='--', marker='', c='k')
    #plt.plot(xi, sample, ls=':', marker='', c='k', label=r"$\sigma/N$ at z=0")
    #plt.plot(xi, -sample, ls=':', marker='', c='k')

    
    inds = (box==0)*(snap==2)
    shot = 1./np.sqrt(N[inds])
    sample = err[inds]/N[inds]
    #xi = x[inds]
    """
    plt.plot(xi, shot, ls='--', marker='', c='gray', label=r"$1/\sqrt{N}$ at z=1")
    plt.plot(xi, -shot, ls='--', marker='', c='gray')
    plt.plot(xi, sample, ls=':', marker='', c='gray', label=r"$\sigma/N$ at z=1")
    plt.plot(xi, -sample, ls=':', marker='', c='gray')
    """

    
    ylim = .1
    plt.ylim(-ylim, ylim)
    axes[0].legend(loc=0, frameon=False, fontsize=12)
    axes[0].set_xlabel(r"$\nu$")
    axes[1].set_xlabel(r"Mass $[{\rm M_\odot}\ h^{-1}]$")
    axes[0].set_ylabel(r"$\frac{N-N_{emu}}{N_{emu}}$")
    axes[1].set_xscale("log")
    
    plt.subplots_adjust(wspace=0, bottom=0.15)#, left=0.20)
    xlim = axes[0].get_xlim()
    axes[0].fill_between(xlim, -0.01, 0.01, color='gray', alpha=0.2)
    axes[0].set_xlim(xlim)
    xlim = axes[1].get_xlim()
    axes[1].fill_between(xlim, -0.01, 0.01, color='gray', alpha=0.2)
    axes[1].set_xlim(xlim)

    fig.set_size_inches(12, 5)
    plt.gcf().savefig("all_residuals.pdf")
    plt.show()
