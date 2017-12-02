"""
This contains the code necessary to predict the Tinker08 MF parameters. 
It does not contain the code for the mass function itself.
"""
import george
import os, inspect
import numpy as np
import scipy.optimize as op
data_path = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))+"/data_files/"
#data_path = "../../fit_mass_functions/output/dfg_rotated/"
R_matrix_path = data_path+"R_matrix.txt"
means_path    = data_path+"rotated_dfg_means.txt"
vars_path     = data_path+"rotated_dfg_vars.txt"
cosmos_path   = data_path+"cosmos.txt"
cosmos = np.genfromtxt(cosmos_path)
cosmos = np.delete(cosmos, 0, 1)  #boxnum
cosmos = np.delete(cosmos, 4, 1)  #ln10As
N_cosmos = len(cosmos)
N_params = len(cosmos[0])

#Pre-computed kernels. These are the results of the commented out section below.
result0 = np.array([0.364223092198, -22.9352155071, 5.51557004041, -7.1216727211, 0.669854126536, 10.3428692548, 26.6530948443, 25.5457824151, -4.7444367057])
result1 = np.array([2.54473246965, -4.55276054632, 5.69753510311, 7.37187134746, 1.18598484046, 12.1132293868, 21.3466119928, 21.0938527045, -0.516388928886])
result2 = np.array([-0.34046637497, -9.82329276108, 35.1445083766, 23.3642668305, 124.534177918, 76.2578943058, 390.515531069, 442.922625002, 83.0533956774])
result3 = np.array([0.668230083534, -11.2523813282, 0.050314037642, 2.26566847213, 2.84085519553, 22.5180516581, 66.8178286442, 74.1828509081, 3.14210708047])
result4 = np.array([0.0329008888935, -12.7083340749, 0.580478721928, -2.29807168398, 23.874797956, -1.12587421992, 68.3917109023, 75.9999309497, 6.14844636968])
result5 = np.array([-0.721871101867, -11.3327038691, 20.9636835002, 0.885361141052, 3.20194263487, 7.09604575815, 62.5469623263, 64.8181871199, 3.29610171114])
result = [result0, result1, result2, result3, result4, result5]

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
            #def nll(p):
            #    gp.set_parameter_vector(p)
            #    ll = gp.lnlikelihood(y, quiet=True)
            #    return -ll if np.isfinite(ll) else 1e25
            #def grad_nll(p):
            #    gp.set_parameter_vector(p)
            #    return -gp.grad_lnlikelihood(y, quiet=True)
            #p0 = gp.get_parameter_vector()
            #results = op.minimize(nll, p0, jac=grad_nll)
            #gp.set_paramter_vector(results.x)
            gp.set_parameter_vector(result[i])
            gplist.append(gp)
        self.gplist = gplist
        return

    def predict_slopes_intercepts(self, cosmo):
        x = np.atleast_2d(cosmo)
        y = self.means.T
        params = np.array([gp.predict(yi, x)[0] for yi,gp in zip(y, self.gplist)])
        return np.dot(self.R, params).flatten()

if __name__ == "__main__":
    t = emu()
    cos = cosmos[0]
    print t.predict_slopes_intercepts(cos)
