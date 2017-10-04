import aemHMF
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    import helper_routines as HR
    scale_factors, zs = HR.get_sf_and_redshifts()


    a = scale_factors[-1]
    M, dndlM = get_darksky_data(a)
    Om = 0.295126
    Ob = 0.0468
    h = 0.688
    sig8 = 0.835
    cosmo = {"om":Om, "ob":Ob, "ol":1-Om, "h":h, "s8":sig8, "ns":0.9676, "w0":-1, "Neff":3.08}
    dndlM_aem = get_prediction(M, a, cosmo)
    pdiff = (dndlM - dndlM_aem)/dndlM_aem

    print (dndlM/dndlM_aem)[:10]

    f, axarr = plt.subplots(2, sharex=True)
    axarr[0].loglog(M, dndlM, ls='', marker='.', c='k')
    axarr[0].loglog(M, dndlM_aem, ls='-', c='b')

    axarr[1].plot(M, pdiff, c='b', ls='-')
    axarr[1].axhline(0, c='k', ls='--')
    plt.show()
