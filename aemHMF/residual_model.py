"""
This contains the model for the residuals
"""
import numpy as np


class residual_model(object):

    def __init__(self):
        """The best fit parameters of the model.
        """
        self.params = np.array([ -2.62986651e+00,   8.20315156e-01,   1.39089166e+00, 2.34282853e-03])


    def predict_residual(self, nu, z):
        a,b,c, floor = self.params
        zx = (1./(1+z))-0.5
        return floor+10**(a+b*(nu-3) + c*zx)


    def covariance_of_residuals(self, nu, z):
        #Note: the covariance length is set to 1.
        sig = self.predict_residual(nu, z)
        C = np.outer(sig, sig)
        for i in range(len(sig)):
            for j in range(i+1, len(sig)):
                dnu = nu[i] - nu[j]
                C[i,j] = C[j,i] = C[i,j]*np.exp(dnu/1.)
        return C

    def residual_realization(self, nu, z):
        C = self.covariance_of_residuals(nu, z)
        return np.random.multivariate_normal(np.zeros_like(nu), C)

if __name__ == "__main__":
    nu = np.linspace(1,6)#, 100)
    z = 0.0
    rmodel = residual_model()
    r = rmodel.predict_residual(nu, z)
    import matplotlib.pyplot as plt
    plt.plot(nu, r, c='b')
    plt.plot(nu, -r, c='b')

    for i in range(5):
        err = rmodel.residual_realization(nu, z)
        plt.plot(nu, err, c='r', alpha=0.5)
    #plt.yscale('log')
    plt.xlabel(r"$\nu$")
    plt.ylabel(r"Error")
    plt.show()
