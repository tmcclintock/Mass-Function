"""
This file contains all of the extra functions that complement the
tinker mass function. Specifically, it contains implementations of
the derivatives with respect to each of the tinker parameters, as
well as calculating the covariance matrix between mass bins.
"""
import numpy as np
from scipy import integrate

"""
Derivatives of the TMF.

'ddX' means d[dndlM]/dX.
"""
def ddd_dndlM(lM, TMF, params):
    M = np.exp(lM)
    sigma = TMF.sigmaM_spline(M)
    d,e,f,g = params
    dgdd = np.exp(-g/sigma**2) * (TMF.dBdd*((sigma/e)**-d+sigma**-f) - TMF.B_coefficient*(sigma/e)**-d)
    return dgdd * TMF.rhom * TMF.deriv_spline(M) #*M/M # log integral

def dde_dndlM(lM, TMF, params):
    M = np.exp(lM)
    sigma = TMF.sigmaM_spline(M)
    d,e,f,g = params
    dgde = np.exp(-g/sigma**2)*(TMF.dBde*((sigma/e)**-d+sigma**-f)-TMF.B_coefficient*d/e*(sigma/e)**-d)
    return dgde * TMF.rhom * TMF.deriv_spline(M) #*M/M #log integral

def ddf_dndlM(lM, TMF, params):
    M = np.exp(lM)
    sigma = TMF.sigmaM_spline(M)
    d,e,f,g = params
    dgdf = TMF.dBdf*((sigma/e)**-d+sigma**-f)*np.exp(-g/sigma**2) - TMF.B_coefficient*sigma**-f*np.log(sigma)*np.exp(-g/sigma**2)
    return dgdf * TMF.rhom * TMF.deriv_spline(M) #*M/M #log integral

def ddg_dndlM(lM, TMF, params):
    M = np.exp(lM)
    sigma = TMF.sigmaM_spline(M)
    d,e,f,g = params
    g_sigma = TMF.B_coefficient*((sigma/e)**-d + sigma**-f) * np.exp(-g/sigma**2)
    dg_sigmadg = -g_sigma/sigma**2 + TMF.dBdg*np.exp(-g/sigma**2)*((sigma/e)**-d + sigma**-f)
    return dg_sigmadg * TMF.rhom * TMF.deriv_spline(M) #*M/M #log integral

"""
Derivatives of the mass function in mass bins.
"""
def derivs_in_bins(TMF, lM_bins, use_numerical_derivatives=False):
    """Compute the derivatives of the mass function in each bin with respect to each tinker parameter.

    Args:
        lM_bins (array_like): List of mass bin edges. Shape must be Nbins by 2. Units are Msun/h.
        use_numerical_derivatives (boolean; optional): Flag to use numerical instead of analytic derivatives. Default is False.

    Returns:
        dndp (array_like): 2D array that is 4 by Nbins of the derivatives of the mass function with respect to each tinker parameter.
    """
    lM_bins = np.log(10**lM_bins)
    dndp = np.zeros((len(lM_bins), 4))
    deriv_functions = [ddd_dndlM, dde_dndlM, ddf_dndlM, ddg_dndlM]
    for i in range(len(lM_bins)):
        lMlow,lMhigh = lM_bins[i]
        for j in range(4):
            if not use_numerical_derivatives: 
                dndp[i,j]= integrate.quad(deriv_functions[j], lMlow, lMhigh, args=(TMF, TMF.params))[0]
            else:
                params_hi   = np.copy(TMF.params)
                params_lo = np.copy(TMF.params)
                Dp = 0.001 * TMF.params[j]
                params_hi[j]   += Dp/2.
                params_lo[j] -= Dp/2.
                TMF.set_parameters(params_hi[0], params_hi[1], params_hi[2], params_hi[3])
                upper = integrate.quad(TMF.dndlM, lMlow, lMhigh)[0]
                TMF.set_parameters(params_lo[0], params_lo[1], params_lo[2], params_lo[3])
                lower = integrate.quad(TMF.dndlM, lMlow, lMhigh)[0]
                dndp[i,j] = (upper-lower)/Dp
    return dndp
