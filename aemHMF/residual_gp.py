"""
This contains the GP for f(nu, z).
"""
import george
import inspect
import numpy as np
import os
data_path = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
p1 = "1p"
data = np.loadtxt(data_path+"/data_files/"+p1+"residuals_bb.txt")
M, nus, zs, R, Re, N,Na,box,snap = data.T

#The kernel for the GP. Hyperparameters have been optimized elsewhere.
lguess = np.array([ 2.41449252, 5.50585975])
#lguess = np.array([ 2.39798486, 2.75768252])
kernel = george.kernels.ExpSquaredKernel(lguess, ndim=len(lguess))

class residual_gp(object):
    
    def __init__(self):
        self.R   = R*0
        self.Re  = Re
        self.gp = george.GP(kernel, white_noise=0.00565521)
        self.x = np.array([nus, zs]).T
        self.gp.compute(self.x, yerr=Re)
    
    def predict_residual(self, nu, z):
        if np.shape(nu) != np.shape(z):
            raise Exception("nu and z must be same dimensions.")
        if not np.shape(nu): #nu and z are scalars
            x = np.atleast_2d([nu, z])
        else: #nu and z are arrays
            x = np.array([nu, z]).T
        return self.gp.predict(self.R, x)

    def residual_realization(self, nu, z):
        if np.shape(nu) != np.shape(z):
            raise Exception("nu and z must be same dimensions.")
        if not np.shape(nu): #nu and z are scalars
            x = np.atleast_2d([nu, z])
        else: #nu and z are arrays
            x = np.array([nu, z]).T
        return self.gp.sample_conditional(self.R, x)


if __name__ == "__main__":
    residual = residual_gp()
    import matplotlib.pyplot as plt
    N = 100
    nu = np.linspace(1, 5.5, N)
    z = np.zeros(N)
    mu, cov = residual.predict_residual(nu, z)
    err = np.sqrt(np.diag(cov))
    plt.fill_between(nu, mu-err, mu+err, color="gray", alpha=0.2)
    for i in range(10):
        plt.plot(nu, residual.residual_realization(nu, z), c='k', alpha=0.2)
    ylim = 0.1
    plt.scatter(nus, R, marker='.', s=1, c='b')
    plt.ylim(-ylim, ylim)
    plt.show()
    
