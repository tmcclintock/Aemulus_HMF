"""
This contains the code necessary to predict the Tinker08 MF parameters. 
It does not contain the code for the mass function itself.
"""
import george
import os, inspect
import numpy as np
import scipy.optimize as op
import aemulus_data as AD
data_path = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))+"/data_files/"
R_matrix_path = data_path+"R.txt"
means_path    = data_path+"r_defg_means.txt"
vars_path     = data_path+"r_defg_vars.txt"
cosmos = AD.get_building_box_cosmologies()
cosmos = cosmos[0:37] #Temporary
cosmos = np.delete(cosmos, -1, 1) #Delete sigma8

N_cosmos = len(cosmos)
N_params = len(cosmos[0])

means  = np.genfromtxt(means_path)
var   = np.genfromtxt(vars_path)
    
class emu(object):

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
            kernel = george.kernels.ExpSquaredKernel(lguess, ndim=N_params)
            gp = george.GP(kernel, mean=np.mean(y), fit_mean=True,
                           white_noise=np.log(np.mean(yerr)**2), fit_white_noise=True)
            gp.compute(cosmos, yerr)
            def nll(p):
                gp.set_parameter_vector(p)
                ll = gp.lnlikelihood(y, quiet=True)
                return -ll if np.isfinite(ll) else 1e25
            def grad_nll(p):
                gp.set_parameter_vector(p)
                return -gp.grad_lnlikelihood(y, quiet=True)
            p0 = gp.get_parameter_vector()
            results = op.minimize(nll, p0, jac=grad_nll)
            gp.set_parameter_vector(results.x)
            gplist.append(gp)
        self.gplist = gplist
        return

    def predict_rotated_params(self, cosmo):
        x = np.atleast_2d(cosmo)
        y = self.means.T
        return np.array([gp.predict(yi, x)[0] for yi, gp in zip(y, self.gplist)])

    def predict_slopes_intercepts(self, cosmo):
        params = self.predict_rotated_params(cosmo)
        return np.dot(self.R, params).flatten()

if __name__ == "__main__":
    t = emu()
    cos = cosmos[0]
    print "ombh2 omch2 w0 ns ln10As H0 Neff\n", cos
    truth = t.means[0]
    err = np.sqrt(t.vars[0])
    pred = t.predict_rotated_params(cos)
    print "Truth Err Pred"
    for i in range(len(truth)):
        print "%.2e %.2e %.2e"%(truth[i], err[i], pred[i])

