"""
Plot the residuals generated.
"""
import numpy as np
import matplotlib.pyplot as plt

zs, lMs, nus, R, Re, box, snap = np.genfromtxt("R_T08.txt", unpack=True)
zsf, lMsf, nusf, Rf, Ref, box, snap = np.genfromtxt("R_f.txt", unpack=True)

plt.plot(nus, R, marker='.', ls='', markersize=1, c='b')
plt.plot(nusf, Rf, marker='.', ls='', markersize=1, c='r')

ylim = .1
plt.ylim(-ylim, ylim)
plt.show()
