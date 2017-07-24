"""
This contains the Tinker08 mass function.
"""

from TMF_extras import *
import cosmocalc as cc
from scipy import special
from scipy import integrate
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
import numpy as np

#Physical constants
G = 4.51701e-48 #Newton's gravitional constant in Mpc^3/s^2/Solar Mass
Mpcperkm = 3.24077927001e-20 #Mpc/km; used to convert H0 to s^-1

class tinker_mass_function(object):
    """A python implementation of the tinker mass function.

    Note: This requires Matt Becker's cosmocalc.
    """

    def __init__(self, cosmo_dict, redshift=0.0, l10M_bounds=[11,16]):
        """Create a TMF_model object.

        Note: the model is created with the default tinker2008_appendix mass function parameters.
        Note: mass units are all assumed to be Msun/h unless otherwise stated.

        Args:
            cosmo_dict (dictionary): Dictionary of cosmological parameters. Only keys necessary to function are "om" and "h" for Omega_m and H0/100.
            redshift (float): Redshift of the mass function; default 0.0.
            l10M_bounds (array_like): Log10 of the upper and lower mass bounds for the splines; defaults to [11,16].
        """
        self.l10M_bounds = np.array(l10M_bounds) #log_10 Mass bounds in Msun/h
        self.redshift = redshift
        self.scale_factor = 1./(1. + self.redshift)
        self.set_new_cosmology(cosmo_dict)
        self.set_parameters(1.97, 1.0, 0.51, 1.228, 0.482)

    def set_parameters(self, d, e, f, g, B=None):
        """Specify the tinker parameters and calculate
        quantities that only depend on them.

        Args:
            d (float): Tinker parameter.
            e (float): Tinker parameter.
            f (float): Tinker parameter.
            g (float): Tinker parameter.
            B (float; optional): Normalization coefficient. If B isn't specified then 
               it's calculated from d,e,f,g such that the mass function is gauranteed 
               to be normalized.
        """
        self.params = np.array([d, e, f, g, B])
        gamma_d2 = special.gamma(d*0.5)
        gamma_f2 = special.gamma(f*0.5)
        log_g = np.log(g)
        gnd2 = g**(-d*0.5)
        gnf2 = g**(-f*0.5)
        ed = e**d
        if not B:
            self.B_coefficient = 2.0/(ed * gnd2 * gamma_d2 + gnf2 * gamma_f2)
            B2 = self.B_coefficient**2
            self.dBdd = 0.25 * B2 * ed * gnd2 * gamma_d2 * (log_g - 2.0 - special.digamma(d*0.5))
            self.dBde = -0.5 * B2 * d * ed/e * gnd2 * gamma_d2
            self.dBdf = 0.25 * B2 * gnf2 * gamma_f2 * (log_g - special.digamma(f*0.5))
            self.dBdg = 0.25 * B2 * (d * ed * gnd2/g * gamma_d2 + f* gnf2/g * gamma_f2)
        else:
            self.B_coefficient = B
            self.dBdd = self.dBde = self.dBdf = self.dBdg = 0
        return

    def set_new_cosmology(self, cosmo_dict):
        """Specify a new set of cosmological parameters and then build splines that depend on these.
        
        Args:
            cosmo_dict (dictionary): Keys are cosmological parameters, specifically om for Omega_matter and h for Hubble constant/100.
        """
        cc.set_cosmology(cosmo_dict)
        Om = cosmo_dict["om"]
        H0 = cosmo_dict["h"]*100.0
        self.rhom=Om*3.*(H0*Mpcperkm)**2/(8*np.pi*G*(H0/100.)**2)#Msunh^2/Mpc^3
        self.cosmo_dict = cosmo_dict
        self.build_splines()
        return

    def build_splines(self):
        """Build the splines needed for integrals over mass bins.
        """
        lM_min,lM_max = self.l10M_bounds
        M_domain = np.logspace(lM_min-1, lM_max+1, num=500)
        sigmaM = np.array([cc.sigmaMtophat_exact(M, self.scale_factor) 
                           for M in M_domain])
        self.sigmaM_spline = IUS(M_domain, sigmaM)
        ln_sig_inv_spline = IUS(M_domain, -np.log(sigmaM))
        deriv_spline = ln_sig_inv_spline.derivative()
        self.deriv_spline = deriv_spline
        return

    def dndlM(self, lM, params=None):
        """Tinker2008_appendix C mass function.

        Args:
            lM (float or array_lke): Ln(Mass) at which to evaluate the mass function.
            params (array_like; optional): the tinker parameters; default is none, in which case it will use
                the parameters already set.

        Returns:
            dndlM (float or array_like): M*dn/dM; the mass function.
        """
        M = np.exp(lM)
        sigma = self.sigmaM_spline(M)
        if params is None: 
            d, e, f, g, B = self.params
        else: 
            d, e, f, g, B = params
            self.set_parameters(d, e, f, g, B)
        g_sigma = self.B_coefficient*((sigma/e)**-d + sigma**-f) * np.exp(-g/sigma**2)
        return g_sigma * self.rhom * self.deriv_spline(M)

    def make_dndlM_spline(self):
        """Creates a spline for dndlM so that the integrals
        over mass bins are faster
        """
        bounds = np.log(10**self.l10M_bounds)
        lM = np.linspace(bounds[0], bounds[1], num=100)
        dndlM = np.array([self.dndlM(lMi) for lMi in lM])
        self.dndlM_spline = IUS(lM, dndlM)
        return lM, dndlM

    def covariance_in_bins(self, lM_bins, Cov_p, use_numerical_derivatives=False):
        """Compute the covariance between each mass bin.
            Args:
                lM_bins (array_like): List of mass bin edges. Shape must be Nbins by 2. Units are Msun/h.
            Cov_p (array_like): Either the variances of the tinker parameters or a matrix with covariances between all tinker parameters.
                use_numerical_derivatives (boolean): Flag to decide how to take the derivatives; default False.

            Returns:
                Cov_NN (array_like): Matrix that is Nbins by Nbins of the covariances between all mass bins.
        """
        dndp = derivs_in_bins(self, lM_bins, use_numerical_derivatives)
        if len(np.shape(Cov_p)) == 1: Cov_p = np.diag(Cov_p)
        cov = np.zeros((len(lM_bins),len(lM_bins)))
        for i in range(len(lM_bins)):
            for j in range(len(lM_bins)):
                cov[i,j] = np.dot(dndp[i], np.dot(Cov_p, dndp[j]))
        return cov

    def n_in_bins(self, lM_bins, redshift=None, params=None):
        """
        IMPORTANT: need to change this funtion. It should switch
        to using a spline for dn/dm and then using
        the integrate function from scipy.


        Compute the tinker mass function in each mass bin.

        Args:
            lM_bins (array_like): List of mass bin edges. Shape must be Nbins by 2. Units are Msun/h.
            redshift (float; optional): Redshift of the mass function. Default is the redshift at initialization.

        Returns:
            n (array_like): Tinker halo mass function at each mass bin. Units are number/ (Mpc/h)^3.
        """
        if redshift is not None:
            if redshift != self.redshift:
                self.redshift = redshift
                self.scale_factor = 1./(1.+redshift)
                self.build_splines()
        lM_bins = np.log(10**lM_bins) #switch to natural log
        return np.array([integrate.quad(self.dndlM,lMlow,lMhigh,args=(params))[0] for lMlow,lMhigh in lM_bins])

#An example of how to use the tinker mass function
if __name__ == "__main__":
    #An example cosmology
    cosmo_dict = {"om":0.3,"ob":0.05,"ol":1.-0.3,"ok":0.0,"h":0.7,"s8":0.77,"ns":0.96,"w0":-1.0,"wa":0.0}

    #Create a TMF object
    TMF = tinker_mass_function(cosmo_dict,redshift=0.0)

    lM = np.log(np.logspace(12, 15, num=20))
    dndlM = TMF.dndlM(lM)

    import matplotlib.pyplot as plt
    plt.loglog(np.exp(lM),dndlM)
    TMF.make_dndlM_spline()
    lM2 = np.log(np.logspace(12, 15, num=3000))
    dndlM2 = np.array([TMF.dndlM_spline(lMi) for lMi in lM2])
    plt.loglog(np.exp(lM2),dndlM2, ls='--')
    plt.xlabel(r"$M\ [h^{-1}{\rm M}_\odot]$",fontsize=24)
    plt.ylabel(r"$n\ [h^3{\rm Mpc^{-3}}]$",fontsize=24)
    plt.subplots_adjust(bottom=0.15, left=0.15)
    plt.title("Mass function at z=0")
    plt.show()
