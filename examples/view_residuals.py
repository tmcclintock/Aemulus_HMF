"""
Plot the residuals generated.
"""
import numpy as np
import matplotlib.pyplot as plt

zs, lMs, nus, R, Re, box, snap = np.genfromtxt("R_T08.txt", unpack=True)
zsf, lMsf, nusf, Rf, Ref, box, snap = np.genfromtxt("R_f.txt", unpack=True)

plt.plot(nus, R, marker='.', ls='', markersize=1, c='b', label=r"$N_{T08}$")
plt.plot(nusf, Rf, marker='.', ls='', markersize=1, c='r', label=r"$N_{T08}(1+f)$")

ylim = .1
plt.ylim(-ylim, ylim)
plt.legend(loc=0, frameon=False)
plt.xlabel(r"$\nu$")
plt.ylabel(r"$\%\ Diff$")
plt.subplots_adjust(bottom=0.15, left=0.15)
plt.show()
