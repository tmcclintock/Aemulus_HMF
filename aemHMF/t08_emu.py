"""
This contains the code necessary to predict the Tinker08 MF parameters. 
It does not contain the code for the mass function itself.
"""
import george
import inspect
import numpy as np
import os
data_path = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))+"/data_files/"
R_matrix_path = data_path+"R_matrix.txt"
means_path    = data_path+"rotated_dfg_means.txt"
vars_path     = data_path+"rotated_dfg_vars.txt"
cosmos_path   = data_path+"cosmos.txt"
cosmos = np.genfromtxt(cosmos_path)
cosmos = np.delete(cosmos, 0, 1)  #boxnum
cosmos = np.delete(cosmos, 4, 1)  #ln10As
cosmos = np.delete(cosmos, -1, 0) #box 39
N_cosmos = len(cosmos)
N_params = len(cosmos[0])

class t08_emu(object):

    def __init__(self):
        self.R      = np.genfromtxt(R_matrix_path)
        self.means  = np.genfromtxt(means_path)
        self.vars   = np.genfromtxt(vars_path)
        self.cosmos = cosmos
        self.train()

    def train(self):
        cosmos = self.cosmos
        lguess = (np.max(cosmos, 0) - np.min(cosmos, 0))/N_cosmos
        means, errs = self.means, np.sqrt(self.vars)
        N_emus = len(means[0])
        gplist = []
        for i in range(N_emus):
            y, yerr = means[:, i], errs[:, i]
            kernel = 1.*george.kernels.ExpSquaredKernel(lguess, ndim=N_params)+george.kernels.WhiteKernel(1, ndim=N_params)
            gp = george.GP(kernel)
            gp.compute(cosmos, yerr)
            gp.optimize(cosmos, y, yerr, verbose=False)
            gplist.append(gp)
        self.gplist = gplist
        return

    def predict_slopes_intercepts(self, cosmo):
        x = np.atleast_2d(cosmo)
        y = self.means.T
        params = np.array([gp.predict(yi, x)[0] for yi,gp in zip(y, self.gplist)])
        return np.dot(self.R, params).flatten()

if __name__ == "__main__":
    t = t08_emu()
    cos = cosmos[0]
    print t.predict_slopes_intercepts(cos)
