"""
This file contains the code needed to, for instance, get the correct parameters,
get the cosmology, get redshifts, etc.
"""
import numpy as np
import matplotlib.pyplot as plt
import sys, os#, emulator

#Paths to the building boxes
base = "../../../all_MF_data/building_data/"#"../Mass-Function-Emulator/test_data/"
datapath = base+"Box%03d/Box%03d_Z%d.txt"
covpath  = base+"Box%03d/Box%03d_cov_Z%d.txt"
def get_basepaths():
    return [base, datapath, covpath]

#Paths to the test boxes
base2 = "../../../all_MF_data/test_data/averaged_mf_data/"
datapath2 = base2+"full_mf_data/TestBox%03d/TestBox%03d_mean_Z%d.txt"
covpath2  = base2+"covariances/TestBox%03d_cov/TestBox%03d_cov_Z%d.txt"
def get_testbox_paths():
    return [base2, datapath2, covpath2]

#Scale factors and redshifts of the sim
scale_factors = np.array([0.25, 0.333333, 0.5, 0.540541, 0.588235, 
                          0.645161, 0.714286, 0.8, 0.909091, 1.0])
redshifts = 1./scale_factors - 1.0

def get_colors(cmapstring="seismic"):
    cmap = plt.get_cmap(cmapstring)
    return [cmap(ci) for ci in np.linspace(1.0, 0.0, N_z)]

def get_sf_and_redshifts():
    return [scale_factors, redshifts]

testbox_cosmos = np.genfromtxt("../aemHMF/data_files/testbox_cosmos.txt")
cosmologies = np.genfromtxt("../aemHMF/data_files/cosmos.txt")

def get_cosmo_dict(index, test_box=False):
    if not test_box:
        num,ombh2,omch2,w0,ns,ln10As,H0,Neff,sigma8 = cosmologies[index]
    else:
        num,ombh2,omch2,w0,ns,ln10As,H0,Neff,sigma8 = testbox_cosmos[index]
    h = H0/100.
    Ob,Om = ombh2/(h**2), ombh2/(h**2)+omch2/(h**2)
    cosmo_dict = {"om":Om, "ob":Ob, "ol":1-Om, "ok":0.0, "h":h, 
                  "s8":sigma8, "ns":ns, "w0":w0, "wa":0.0, "Neff":Neff}
    return cosmo_dict

def get_building_cosmos(remove_As=True):
    building_cosmos = np.delete(cosmologies, 0, 1) #Delete boxnum
    if remove_As: building_cosmos = np.delete(building_cosmos, 4, 1)
    return building_cosmos

def get_testbox_cosmos(remove_As=True):
    tb_cosmos = testbox_cosmos
    if remove_As: tb_cosmos = np.delete(testbox_cosmos, 4, 1) #Delete ln10As
    return tb_cosmos

#Routines for getting sim data
def get_sim_data(sim_index, z_index):
    base, datapath, covpath = get_basepaths()
    data = np.loadtxt(datapath%(sim_index, sim_index, z_index))
    lM_bins = data[:,:2]
    lM = np.mean(lM_bins, 1)
    N = data[:,2]
    cov = np.loadtxt(covpath%(sim_index, sim_index, z_index))
    err = np.sqrt(np.diagonal(cov))
    return lM_bins, lM, N, err, cov

def get_testbox_data(sim_index, z_index):
    base, datapath, covpath = get_testbox_paths()
    data = np.loadtxt(datapath%(sim_index, sim_index, z_index))
    N = data[:,2]
    goodinds = N>0
    data = data[goodinds]
    lM_bins = data[:,:2]
    lM = np.mean(lM_bins, 1)
    N = data[:,2]
    cov = np.loadtxt(covpath%(sim_index, sim_index, z_index))
    cov = cov[goodinds]
    cov = cov[:,goodinds]
    err = np.sqrt(np.diagonal(cov))
    return lM_bins, lM, N, err, cov
