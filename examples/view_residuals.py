"""
Plot the residuals generated.
"""
import numpy as np
import matplotlib.pyplot as plt
import aemulus_data as AD
import aemHMF

def get_resids(path, use_nu=True):
    return np.genfromtxt(path, unpack=True)


if __name__ == "__main__":
    use_nu = True
    zs, lMs, nus, R, Re, N, err, Nt08, box, snap = get_resids("BB_residuals.txt", use_nu)
    if use_nu: x = nus
    else: x = lMs
    plt.plot(x, R, marker='.', ls='', markersize=1, c='b', label="Building boxes", alpha=0.2)

    zs, lMs, nus, R, Re, N, err, Nt08, box, snap = get_resids("test_residuals.txt", use_nu)
    if use_nu: x = nus
    else: x = lMs
    plt.plot(x, R, marker='.', ls='', markersize=2, c='r', label="Test boxes", alpha=1)
    

    plt.axhline(0, ls='-', c='k')
    inds = (box==0)*(snap==9)
    sample_variance = 1./np.sqrt(N[inds])
    xi = x[inds]
    plt.plot(xi, sample_variance, ls='--', marker='', c='k', label=r"$1/\sqrt{N}$ at z=0")
    plt.plot(xi, -sample_variance, ls='--', marker='', c='k')
    
    inds = (box==0)*(snap==2)
    print snap
    sample_variance = 1./np.sqrt(N[inds])
    xi = x[inds]
    plt.plot(xi, sample_variance, ls=':', marker='', c='k', label=r"$1/\sqrt{N}$ at z=1")
    plt.plot(xi, -sample_variance, ls=':', marker='', c='k')

    
    ylim = .1
    plt.ylim(-ylim, ylim)
    plt.legend(loc=0, frameon=False)
    if use_nu: plt.xlabel(r"$\nu$")
    else: plt.xlabel(r"$\log_{10}M$")
    plt.ylabel(r"$\%\ Diff$")
    plt.subplots_adjust(bottom=0.15, left=0.15)
    xlim = plt.gca().get_xlim()
    plt.fill_between(xlim, -0.01, 0.01, color='gray', alpha=0.2)
    plt.xlim(xlim)
    plt.show()
