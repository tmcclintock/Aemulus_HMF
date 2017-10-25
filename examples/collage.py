"""
Get all the snapshots from all building boxes and plot them ALL.
"""
import aemHMF
import numpy as np
import matplotlib.pyplot as plt
import aemulus_data as AD
plt.rc("text", usetex=True)
plt.rc("font", size=18)
plt.rc('font', family='serif')

Nboxes = 40
Nsnaps = 10

if __name__ == "__main__":

    sfs = AD.get_scale_factors()
    zs = 1./sfs - 1
    colors = [plt.get_cmap("seismic")(ci) for ci in np.linspace(1.0, 0.0, len(sfs))]

    for box in range(Nboxes):
        for snapshot in range(Nsnaps):
            a = sfs[snapshot]
            z = zs[snapshot]
            path = AD.path_to_building_box_data(box, snapshot)
            lMlo, lMhi, N, Mtot = np.genfromtxt(path, unpack=True)
            M_bins = 10**np.array([lMlo, lMhi]).T
            M = Mtot/N
            if box == 0: plt.loglog(M, N, c=colors[snapshot], label=r"$z=%.2f$"%z)
            else: plt.loglog(M, N, c=colors[snapshot], alpha=0.2)
        print "Box %d"%box

    plt.legend(loc=0, frameon=0, fontsize=10)
    plt.xlabel(r"Mass $[{\rm M_\odot} h^{-1}]$")
    plt.ylabel(r"$\Delta N/N_{emu}$")
    plt.ylabel(r"Number")
    plt.ylim(1, 5e5)
    xlim = plt.gca().get_xlim()
    plt.fill_between(xlim, -0.01, 0.01, color="gray", alpha=0.2)
    plt.xlim(xlim)
    plt.subplots_adjust(bottom=0.15)
    plt.gcf().savefig("collage_figure.pdf")
    plt.show()
