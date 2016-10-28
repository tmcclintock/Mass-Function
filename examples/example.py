import tinker_mass_function as TMF
import numpy as np
#An example cosmology
cosmo_dict = {"om":0.3,"ob":0.05,"ol":1.-0.3,\
                  "ok":0.0,"h":0.7,"s8":0.77,\
                  "ns":3.0,"w0":.96,"wa":0.0}
redshift = 0.0

MF = TMF.MF_model(cosmo_dict,redshift)

d,e,f,g = 1.97,1.0,0.51,1.228 #d,e,f,g
MF.set_parameters(d,e,f,g)

lM_bins = MF.make_bins(Nbins=20,lM_low=12,lM_high=15)
Masses = np.mean(10**lM_bins,1) #The mid points of the bins, just for plotting
n_z0 = MF.n_in_bins(lM_bins)
n_z1 = MF.n_in_bins(lM_bins,1.0)

import matplotlib.pyplot as plt
plt.loglog(Masses,n_z0,label=r"$z=0$")
plt.loglog(Masses,n_z1,label=r"$z=1$")
plt.ylabel(r"$n\ [h^3{\rm Mpc^{-3}}]$")
plt.xlabel(r"$M\ [h^{-1}{\rm M}_\odot]$")
plt.legend()
plt.show()
