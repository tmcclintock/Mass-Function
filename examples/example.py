import tinker_mass_function as TMF
import numpy as np
#An example cosmology
cosmo_dict = {"om":0.3,"ob":0.05,"ol":1.-0.3,
              "ok":0.0,"h":0.7,"s8":0.77,
              "ns":3.0,"w0":.96,"wa":0.0}
redshift = 0.0

example_TMF = TMF.TMF_model(cosmo_dict,redshift)

d,e,f,g = 1.97,1.0,0.51,1.228 #d,e,f,g
example_TMF.set_parameters(d,e,f,g)

lM_bins = example_TMF.make_bins(Nbins=20,lM_low=12,lM_high=15)
Masses = np.mean(10**lM_bins,1) #The mid points of the bins, just for plotting
n_z0 = example_TMF.n_in_bins(lM_bins)
Cov_p = np.diag(np.ones((4))*0.01)
n_cov_z0 =example_TMF.covariance_in_bins(lM_bins,Cov_p)
n_err_z0 = np.sqrt(np.diagonal(n_cov_z0))
n_z1 = example_TMF.n_in_bins(lM_bins,redshift=1.0)
n_cov_z1 =example_TMF.covariance_in_bins(lM_bins,Cov_p)
n_err_z1 = np.sqrt(np.diagonal(n_cov_z1))

import matplotlib.pyplot as plt
plt.errorbar(Masses,n_z0,n_err_z0,label=r"$z=0$")
plt.errorbar(Masses,n_z1,n_err_z1,label=r"$z=1$")
plt.xscale('log')
plt.yscale('log')
plt.ylabel(r"$n\ [h^3{\rm Mpc^{-3}}]$")
plt.xlabel(r"$M\ [h^{-1}{\rm M}_\odot]$")
plt.legend()
plt.show()
