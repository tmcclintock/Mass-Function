"""
This contains the mass function object.

Dependencies: numpy, scipy, cosmocalc
"""

import cosmocalc as cc
from scipy import special
from scipy import integrate
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
import numpy as np

TOL = 1e-3

class Mass_Function(object):
    """
    This function takes in low and high log_10 mass
    values and creates bins (i.e. an Nbinsx2 array 
    where Nbins is the number of bins).
    """
    def make_bins(self,Nbins,lM_low,lM_high):
        lM_edges = np.linspace(lM_low,lM_high,Nbins+1)
        return np.array(zip(lM_edges[:-1],lM_edges[1:]))
    
    """
    Initialization function. It requires a dictionary
    that contains the cosmology to be passed to
    cosmocalc, the upper and lower bounds of the 
    mass in log_10 space, the
    volume we are calculating the mass function in,
    and the redshift we are working at.
    """
    def __init__(self,cosmo_dict,lM_bounds,volume,redshift):
        self.lM_bounds = lM_bounds
        self.volume = volume #volume of the simulation
        self.redshift = redshift
        self.scale_factor = 1./(1+self.redshift)
        self.prev_scale_factor = self.scale_factor
        self.set_new_cosmology(cosmo_dict,self.scale_factor)

    """
    This function is for if one wants to use a 
    different cosmology. It passes a new cosmology
    to cosmocalc and then rebuilds the sigma splines.
    """
    def set_new_cosmology(self,cosmo_dict,scale_factor):
        #Constants
        G = 4.52e-48 #Newton's gravitional constant in Mpc^3/s^2/Solar Mass
        Mpcperkm = 3.241e-20 #Mpc/km; used to convert H0 to s^-1
        self.cosmo_dict = cosmo_dict
        cc.set_cosmology(cosmo_dict)
        self.build_splines(scale_factor)
        Om,H0 = cosmo_dict["om"],cosmo_dict["h"]*100.0
        self.rhom=Om*3.*(H0*Mpcperkm)**2/(8*np.pi*G*(H0/100.)**2)#Msun h^2/Mpc^3

    def build_splines(self,scale_factor):
        lM_min,lM_max = self.lM_bounds
        M_space = np.logspace(lM_min-1,lM_max+1,500,base=10)
        sigmaM = np.array([cc.sigmaMtophat_exact(M,scale_factor)\
                           for M in M_space])
        ln_sig_inv_spline = IUS(M_space,-np.log(sigmaM))
        deriv_spline = ln_sig_inv_spline.derivative()
        self.deriv_spline = deriv_spline
        self.splines_built = True
        return

    def B_coeff(self,d,e,f,g):
        return 2.0/(e**d * g**(-d/2.)*special.gamma(d/2.) + g**(-f/2.)*special.gamma(f/2.))
        
    def calc_g(self,sigma,params):
        d,e,f,g = params
        return self.B_coeff(d,e,f,g)*((sigma/e)**-d + sigma**-f) * np.exp(-g/sigma**2)

    """
    This is a wrapper for sigmaMtophat.
    """
    def sigmaMtophat(self,M,scale_factor):
        return cc.sigmaMtophat(M,scale_factor)

    def dndM_at_M(self,lM,params):
        rhom,dln_sig_inv_dM_spline = self.rhom,self.deriv_spline
        M = np.exp(lM)
        g_sigma = self.calc_g(self.sigmaMtophat(M,self.scale_factor),params)
        return g_sigma * self.rhom*dln_sig_inv_dM_spline(M) #*M/M #log integral

    def Mass_Function_in_bin(self,lMlow,lMhigh,params):
        return integrate.quad(self.dndM_at_M,lMlow,lMhigh,args=(params),\
                              epsabs=TOL,epsrel=TOL/10.)[0]*self.volume
        
    def Mass_Function_all_bins(self,lM_bins,params,redshift):
        if self.redshift != redshift:
            self.redshift = redshift
            self.scale_factor = 1./(1.+redshift)
            self.build_splines(self.scale_factor)
        return np.array([self.Mass_Function_in_bin(lMlow,lMhigh,params) for lMlow,lMhigh in lM_bins])

if __name__ == "__main__":
    #An example cosmology
    cosmo_dict = {"om":0.3,"ob":0.05,"ol":1.-0.3,\
                  "ok":0.0,"h":0.7,"s8":0.77,\
                  "ns":3.0,"w0":.96,"wa":0.0}
    bounds = np.log10([1e12,1e16]) #Mass bounds in Msun/h
    volume = 1e9 #(Mpc/h)^3
    redshift = 0.0

    MF = Mass_Function(cosmo_dict,bounds,volume,redshift)

    params = np.array([1.97,1.0,0.51,1.228]) #d,e,f,g

    bins = MF.make_bins(np.log(1e10),np.log(1e12),10)
    output = MF.Mass_Function_all_bins(bins,params,redshift)

    Masses = np.mean(bins,1)
    import matplotlib.pyplot as plt
    plt.plot(Masses,output)
    plt.show()
