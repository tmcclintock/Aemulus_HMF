"""
This file contains the actual Tinker08 mass function. It uses t08_emu to get the mass function parameters.
"""
#Public modules
import cosmocalc as cc
import numpy as np
from scipy import special, integrate
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
#From this directory
import t08_emu as emu

#Physical constants
G = 4.51701e-48 #Newton's gravitional constant in Mpc^3/s^2/Solar Mass
Mpcperkm = 3.24077927001e-20 #Mpc/km; used to convert H0 to s^-1
rhocrit = 3.*(Mpcperkm*100)**2/(8*np.pi*G) #Msun h^2/Mpc^3

class n_t08(object):
    
    def __init__(self, cosmo_dict):
        self.cosmo_dict = cosmo_dict
        #get parameters from t08_emu
        #self.calc_special_functions()
        #self.build_splines()

    def built_splines(self):
        

if __name__ == "__main__":
    print "working"
