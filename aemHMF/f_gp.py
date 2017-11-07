"""
This contains the GP for f(nu, z).
"""
import george
import inspect
import numpy as np
import os
data_path = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
#Rt08_path = data_path+"/data_files/R_T08_residuals.txt"
Rt08_path = data_path+"/data_files/R_T08.txt"
zs, lMs, nus, R, Re, box, snap = np.genfromtxt(Rt08_path, unpack=True)

#GP_parameters, computed ahead of time.
k = george.kernels.ConstantKernel(log_constant= -7.4627695322, ndim=2, axes=np.array([0, 1]))
metric = [ 0.45806604,  1.2785944 ]

class f_gp(object):
    
    def __init__(self):
        z, lM, nu, R, eR, box_inds, snap_inds = np.genfromtxt(Rt08_path, unpack=True)
        self.z   = z
        self.nus = nu
        self.R   = R
        self.eR  = eR
        kernel = k*george.kernels.ExpSquaredKernel(metric, ndim=2)
        self.gp = george.GP(kernel)
        self.x = np.array([nu, z]).T
        self.gp.compute(self.x, yerr=self.eR)
    
    def predict_f(self, nu, z):
        if np.shape(nu) != np.shape(z):
            raise Exception("nu and z must be same dimensions.")
        if not np.shape(nu): #nu and z are scalars
            x = np.atleast_2d([nu, z])
        else: #nu and z are arrays
            x = np.array([nu, z]).T
        return self.gp.predict(self.R, x)

    def sample_f(self, nu, z):
        if np.shape(nu) != np.shape(z):
            raise Exception("nu and z must be same dimensions.")
        if not np.shape(nu): #nu and z are scalars
            x = np.atleast_2d([nu, z])
        else: #nu and z are arrays
            x = np.array([nu, z]).T
        return self.gp.sample_conditional(self.R*0, x)


if __name__ == "__main__":
    f = f_gp()
    import matplotlib.pyplot as plt
    nu = np.linspace(0, 7, 100)
    z = np.zeros(100)
    mu, cov = f.predict_f(nu, z)
    err = np.sqrt(np.diag(cov))
    plt.plot(nu, mu)
    plt.fill_between(nu, mu-err, mu+err,alpha=0.4)
    plt.plot(nu, f.sample_f(nu, z), c='r')
    ylim = 0.1
    plt.plot(nus, R, marker='.', ls='', markersize=1)
    plt.ylim(-ylim, ylim)
    plt.show()
    
