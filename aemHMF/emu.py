"""
This contains the code necessary to predict the Tinker08 MF parameters. 
It does not contain the code for the mass function itself.
"""
import george
from george.kernels import *
import os, inspect
import numpy as np
import scipy.optimize as op
import aemulus_data as AD
data_path = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))+"/data_files/"

model = "np6_mi5"
p1 = ""
R_matrix_path = data_path+p1+"R_%s.txt"%model
means_path    = data_path+p1+"r_%s_means.txt"%model
true_means_path = data_path+"%s_means.txt"%model
vars_path     = data_path+p1+"r_%s_vars.txt"%model
cosmos = AD.building_box_cosmologies()
cosmos = np.delete(cosmos, -1, 1) #Delete sigma8

N_cosmos = len(cosmos)
N_params = len(cosmos[0])

means  = np.genfromtxt(means_path)
var   = np.genfromtxt(vars_path)
truemeans = np.loadtxt(true_means_path)
R = np.loadtxt(R_matrix_path)
    
class emu(object):

    def __init__(self):
        self.R      = R
        self.means  = means
        self.vars   = var
        self.cosmos = cosmos
        self.train()
        self.name=model

    def train(self):
        cosmos = self.cosmos
        lguess = np.std(cosmos,0)
        means, errs = self.means, np.sqrt(self.vars)
        N_emus = len(means[0])
        gplist = []
        for i in range(N_emus):
            y, yerr = means[:, i], errs[:, i]
            kernel = ExpSquaredKernel(lguess, ndim=N_params)
            gp = george.GP(kernel, mean=np.mean(y), fit_mean=False,
                           white_noise=np.log(np.mean(yerr)**2), fit_white_noise=False)
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
            #print results.x
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
    box = 26
    t = emu()
    cos = cosmos[box]
    print "ombh2 omch2 w0 ns ln10As H0 Neff\n", cos
    truth = t.means[box]
    err = np.sqrt(t.vars[box])
    pred = t.predict_rotated_params(cos)
    print "Rotated Parameters:"
    print "Truth   \tErr\t        Pred"
    for i in range(len(truth)):
        print "%.2e\t%.2e\t%.2e"%(truth[i], err[i], pred[i])
    print "Unrotated Parameters:"
    pred = t.predict_slopes_intercepts(cos)
    truth = np.dot(R, truth).flatten()
    for i in range(len(truth)):
        print "%.2e\t%.2e"%(truth[i], pred[i])
