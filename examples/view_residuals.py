"""
Plot the residuals generated.
"""
import numpy as np
import matplotlib.pyplot as plt

def get_resids(path, use_nu=True):
    zs, lMs, nus, R, Re, box, snap = np.genfromtxt(path, unpack=True)
    if use_nu: return nus, R, Re, zs
    else: return lMs, R, Re, zs


if __name__ == "__main__":
    use_nu = True
    x, R, Re, zs = get_resids("BB_residuals.txt", use_nu)
    plt.plot(x, R, marker='.', ls='', markersize=1, c='b', label="Building boxes", alpha=0.2)

    x, R, Re, zs = get_resids("test_residuals.txt", use_nu)
    plt.plot(x, R, marker='.', ls='', markersize=2, c='r', label="Test boxes", alpha=1)

    plt.axhline(0, ls='-', c='k')
    
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