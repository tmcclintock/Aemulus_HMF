import pytest
import numpy as np
import numpy.testing as npt
from aemHMF import tinkerMF as tmf

cos = {"Omega_m":0.3,
       "Omega_b":0.05,
       "h":0.7,
       "sigma8":0.77,
       "n_s":0.96,
       "w0":-1.0,
       "N_eff":3.0,
       "wa":0.0} #This is a requirement

mf = tmf.tinkerMF()
mf.set_cosmology(cos)
M = np.logspace(12, 16, num=1000)
Mbins = np.logspace(12, 16, num=15)
Mbins = np.array([Mbins[:-1], Mbins[1:]]).T
sfs = np.array([0.25, 0.3333333, 0.5, 1.0])
zs = 1./sfs - 1.

def test_tinkerMF():
    assert hasattr(tmf, 'tinkerMF')
    assert hasattr(mf, 'set_cosmology')
    assert hasattr(mf, 'cos_from_dict')
    assert hasattr(mf, 'calc_normalization')
    assert hasattr(mf, 'merge_t08_params')
    assert hasattr(mf, 'sigmaM')
    assert hasattr(mf, 'Gsigma')
    assert hasattr(mf, 'GM')
    assert hasattr(mf, 'dndlM')
    assert hasattr(mf, 'n_bin')
    assert hasattr(mf, 'n_bins')
    
def test_dndlm_mass_dependence():
    for a in zs:
        dndlM = np.array([mf.dndlM(Mi, a) for Mi in M])
        npt.assert_array_less(dndlM[1:], dndlM[:-1])

def test_bin_mass_dependence():
    for a in zs:
        n = mf.n_bins(Mbins, a)
        npt.assert_array_less(n[1:], n[:-1])

def test_bin_vs_bins():
    for a in zs:
        narr = mf.n_bins(Mbins, a)
        narr2 = np.array([mf.n_bin(Mbi[0], Mbi[1], a) for Mbi in Mbins])
        npt.assert_array_equal(narr, narr2)

def test_sf_dependence():
    for i in range(len(zs)-1):
        n1 = mf.n_bins(Mbins, zs[i]) #higher z
        n2 = mf.n_bins(Mbins, zs[i+1]) #lower z
        npt.assert_array_less(n1/n1, n2/n1)
