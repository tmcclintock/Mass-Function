"""
This contains the Tinker08 mass function.
"""

import cosmocalc as cc
from scipy import special
from scipy import integrate
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
import numpy as np

TOL = 1e-3

class MF_model(object):
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
    cosmocalc, and the redshift we are working at.

    Optionally you can pass in log_10 of the mass bounds
    to build the spline between.
    """
    def __init__(self,cosmo_dict,redshift,l10M_bounds=[11,16],use_numerical_derivatives=False):
        self.l10M_bounds = l10M_bounds #log_10 Mass bounds in Msun/h
        self.redshift = redshift
        self.scale_factor = 1./(1.+self.redshift)
        self.set_new_cosmology(cosmo_dict)
        self.params_are_set = False
        self.use_numerical_derivatives = use_numerical_derivatives

    def set_parameters(self,d,e,f,g):
        """
        Specify the tinker parameters and calculate quantities
        that only depend on them.
        """

        self.d = d
        self.e = e
        self.f = f
        self.g = g
        self.params = np.array([self.d,self.e,self.f,self.g])
        self.B_coefficient = 2.0/(e**d * g**(-d/2.)*special.gamma(d/2.) + g**(-f/2.)*special.gamma(f/2.))
        self.dBdd = self.B_coefficient**2/4.*e**d*g**(-d/2.)*special.gamma(d/2.)*(np.log(g)-2-special.digamma(d/2.))
        self.dBde = self.B_coefficient**2/2.*(-d)*e**(d-1)*g**(-d/2.)*special.gamma(d/2)
        self.dBdf = self.B_coefficient**2/4.*g**(-f/2.)*special.gamma(f/2.)*(np.log(g)-special.digamma(f/2.))
        self.dBdg = self.B_coefficient**2/4.*(d*e**d*g**(-d/2.-1)*special.gamma(d/2.)+f*g**(-f/2.-1)*special.gamma(f/2.))
        self.params_are_set = True

    def set_new_cosmology(self,cosmo_dict):
        G = 4.52e-48 #Newton's gravitional constant in Mpc^3/s^2/Solar Mass
        Mpcperkm = 3.241e-20 #Mpc/km; used to convert H0 to s^-1
        cc.set_cosmology(cosmo_dict) #Used to create the splines in cosmocalc
        Om,H0 = cosmo_dict["om"],cosmo_dict["h"]*100.0
        self.rhom=Om*3.*(H0*Mpcperkm)**2/(8*np.pi*G*(H0/100.)**2)#Msunh^2/Mpc^3
        self.cosmo_dict = cosmo_dict
        self.build_splines()

    def build_splines(self):
        lM_min,lM_max = self.l10M_bounds
        M_domain = np.logspace(lM_min-1,lM_max+1,500,base=10)
        sigmaM = np.array([cc.sigmaMtophat_exact(M,self.scale_factor) for M in M_domain])
        ln_sig_inv_spline = IUS(M_domain,-np.log(sigmaM))
        deriv_spline = ln_sig_inv_spline.derivative()
        self.deriv_spline = deriv_spline
        self.splines_built = True

    """
    The following functions are only used in 
    calculating the variance in a mass bin.
    """
    def ddd_dndlM_at_lM(self,lM,params):
        M = np.exp(lM)
        sigma = cc.sigmaMtophat(M,self.scale_factor)
        d,e,f,g = params
        dgdd = np.exp(-g/sigma**2)*(self.dBdd*((sigma/e)**-d+sigma**-f)-self.B_coefficient*(sigma/e)**-d)
        return dgdd * self.rhom * self.deriv_spline(M) #*M/M #log integral

    def dde_dndlM_at_lM(self,lM,params):
        M = np.exp(lM)
        sigma = cc.sigmaMtophat(M,self.scale_factor)
        d,e,f,g = params
        dgde = np.exp(-g/sigma**2)*(self.dBde*((sigma/e)**-d+sigma**-f)-self.B_coefficient*d/e*(sigma/e)**-d)
        return dgde * self.rhom * self.deriv_spline(M) #*M/M #log integral

    def ddf_dndlM_at_lM(self,lM,params):
        M = np.exp(lM)
        sigma = cc.sigmaMtophat(M,self.scale_factor)
        d,e,f,g = params
        dgdf = self.dBdf*((sigma/e)**-d+sigma**-f)*np.exp(-g/sigma**2) - self.B_coefficient*sigma**-f*np.log(sigma)*np.exp(-g/sigma**2)
        return dgdf * self.rhom * self.deriv_spline(M) #*M/M #log integral

    def ddg_dndlM_at_lM(self,lM,params):
        M = np.exp(lM)
        sigma = cc.sigmaMtophat(M,self.scale_factor)
        d,e,f,g = params
        g_sigma = self.B_coefficient*((sigma/e)**-d + sigma**-f) * np.exp(-g/sigma**2)
        dg_sigmadg = -g_sigma/sigma**2 + self.dBdg*np.exp(-g/sigma**2)*((sigma/e)**-d + sigma**-f)
        return dg_sigmadg * self.rhom * self.deriv_spline(M) #*M/M #log integral

    """
    The following contain the
    tinker mass function.
    """
    def dndlM_at_lM(self,lM,params):
        M = np.exp(lM)
        sigma = cc.sigmaMtophat(M,self.scale_factor)
        d,e,f,g = params
        g_sigma = self.B_coefficient*((sigma/e)**-d + sigma**-f) * np.exp(-g/sigma**2)
        return g_sigma * self.rhom * self.deriv_spline(M)

    """
    This is the analytic derivative of n 
    for each bin with respect to f and g.
    """
    def derivs_in_bins(self,lM_bins):
        lM_bins = np.log(10**lM_bins)
        dndf,dndg = [],[]
        for lMlow,lMhigh in lM_bins:
            if not self.use_numerical_derivatives:
                dndf.append(integrate.quad(self.ddf_dndlM_at_lM,lMlow,lMhigh,args=(self.params))[0])
                dndg.append(integrate.quad(self.ddg_dndlM_at_lM,lMlow,lMhigh,args=(self.params))[0])
            else:
                params = np.copy(self.params)
                Df = 0.1
                Dg = 0.1
                self.set_parameters(params[0],params[1],params[2]+Df/2.,params[3])
                upper = integrate.quad(self.dndlM_at_lM,lMlow,lMhigh,args=(self.params))[0]
                self.set_parameters(params[0],params[1],params[2]-Df/2.,params[3])
                lower = integrate.quad(self.dndlM_at_lM,lMlow,lMhigh,args=(self.params))[0]
                dndf.append((upper-lower)/Df)
                self.set_parameters(params[0],params[1],params[2],params[3]+Dg/2.)
                upper = integrate.quad(self.dndlM_at_lM,lMlow,lMhigh,args=(self.params))[0]
                self.set_parameters(params[0],params[1],params[2],params[3]-Dg/2.)
                lower = integrate.quad(self.dndlM_at_lM,lMlow,lMhigh,args=(self.params))[0]
                dndg.append((upper-lower)/Dg)
                self.set_parameters(params[0],params[1],params[2],params[3])
        #Still need to propagate the derivatives here.
        return np.array([dndf,dndg])

    """
    This is the variance in all bins.
    The variable 'variances' contains the
    variance in the paramters, namely f and g.
    """
    def variance_in_bins(self,lM_bins,variances):
        dndf,dndg = self.derivs_in_bins(lM_bins)
        return dndf**2*variances[0] + dndg**2*variances[1]

    """
    Similar to the above, this gives the full
    covariance matrix between the bins.
    'Variances' contain the variance of f
    and then the variance of g.
    """
    def covariance_in_bins(self,lM_bins,variances,fg_covariance=0):
        #Still need to propagate derivatives here.
        dndf,dndg = self.derivs_in_bins(lM_bins)
        cov = np.zeros((len(lM_bins),len(lM_bins)))
        for i in range(len(lM_bins)):
            cov[i] = dndf[i]*dndf*variances[0] + dndg[i]*dndg*variances[1] + fg_covariance*(dndf[i]*dndg + dndg[i]*dndf)
            continue #end i
        return cov

    """
    Returns the number density [#/(Mpv/h)^3]
    of halos within the mass bins.
    """
    def n_in_bins(self,lM_bins,redshift=None):
        if not self.params_are_set: raise ValueError("Must set parameters before modeling.")
        if redshift is not None:
            if redshift != self.redshift:
                self.redshift = redshift
                self.scale_factor = 1./(1.+redshift)
                self.build_splines()
        lM_bins = np.log(10**lM_bins)
        return np.array([integrate.quad(self.dndlM_at_lM,lMlow,lMhigh,args=(self.params),epsabs=TOL,epsrel=TOL/10.)[0] for lMlow,lMhigh in lM_bins])

    def n_approx_in_bins(self,lM_bins,redshift=None):
        if not self.params_are_set: raise ValueError("Must set parameters before modeling.")
        if redshift is not None:
            if redshift != self.redshift:
                self.redshift = redshift
                self.scale_factor = 1./(1.+redshift)
                self.build_splines()
        lM_bins = np.log(10**lM_bins)
        answer = np.array([0.5*(lM_high-lM_low)*(self.dndlM_at_lM(lM_high,self.params)+self.dndlM_at_lM(lM_low,self.params)) for lM_low,lM_high in lM_bins])
        return answer

if __name__ == "__main__":
    #An example cosmology
    cosmo_dict = {"om":0.3,"ob":0.05,"ol":1.-0.3,\
                  "ok":0.0,"h":0.7,"s8":0.77,\
                  "ns":3.0,"w0":.96,"wa":0.0}

    MF = MF_model(cosmo_dict,redshift=0.0)

    d,e,f,g = 1.97,1.0,0.51,1.228 #d,e,f,g
    MF.set_parameters(d,e,f,g)

    lM_bins = MF.make_bins(Nbins=10,lM_low=12,lM_high=15)
    Masses = np.mean(10**lM_bins,1) #The mid points of the bins, just for plotting
    n_z0 = MF.n_in_bins(lM_bins)
    n_var_z0 = MF.variance_in_bins(lM_bins,[0.01,0.01])
    n_z1 = MF.n_in_bins(lM_bins,1.0)

    print MF.derivs_in_bins(lM_bins)

    import matplotlib.pyplot as plt
    plt.loglog(Masses,n_z1,label=r"$z=1$")
    plt.errorbar(Masses,n_z0,np.sqrt(n_var_z0),label=r"$z=0$")
    plt.ylabel(r"$n\ [h^3{\rm Mpc^{-3}}]$")
    plt.xlabel(r"$M\ [h^{-1}{\rm M}_\odot]$")
    plt.legend()
    plt.show()
