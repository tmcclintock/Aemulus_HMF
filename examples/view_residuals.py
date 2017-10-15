"""
Plot the residuals generated.
"""
import numpy as np
import matplotlib.pyplot as plt

zs, lMs, nus, R, Re, box, snap = np.genfromtxt("R_T08.txt", unpack=True)

plt.plot(nus, R, marker='.', ls='', markersize=1)
ylim = .1
plt.ylim(-ylim, ylim)
plt.show()
