import tinker_mass_function as TMF
import numpy as np
import matplotlib.pyplot as plt

#An example cosmology
cosmo_dict = {"om":0.3,"ob":0.05,"ol":1.-0.3,"ok":0.0,"h":0.7,"s8":0.77,"ns":0.96,"w0":-1.0,"wa":0.0}
zs = np.array([0.0, 1.0])

lM = np.logspace(12, 15, num=21)
lM_bins = np.log10(np.array([lM[:-1], lM[1:]]).T)
Masses = np.mean(10**lM_bins,1) #The mid points of the bins, just for plotting

for z in zs:
    example_TMF = TMF.tinker_mass_function(cosmo_dict, redshift=z)
    n = example_TMF.n_in_bins(lM_bins)
    Cov_p = np.diag(np.ones((4))*0.01)
    n_cov =example_TMF.covariance_in_bins(lM_bins, Cov_p)
    n_err = np.sqrt(np.diagonal(n_cov))
    plt.errorbar(Masses,n,n_err,label=r"$z=%.1f$ no NDs"%z)

for z in zs:
    example_TMF = TMF.tinker_mass_function(cosmo_dict, redshift=z)
    n = example_TMF.n_in_bins(lM_bins)
    Cov_p = np.diag(np.ones((4))*0.01)
    n_cov =example_TMF.covariance_in_bins(lM_bins, Cov_p, True)
    n_err = np.sqrt(np.diagonal(n_cov))
    plt.errorbar(Masses,n,n_err,label=r"$z=%.1f$ yes NDs"%z)

plt.xscale('log')
plt.yscale('log')
plt.ylabel(r"$n\ [h^3{\rm Mpc^{-3}}]$")
plt.xlabel(r"$M\ [h^{-1}{\rm M}_\odot]$")
plt.legend(loc=0)
plt.show()
