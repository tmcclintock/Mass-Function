"""
This file contains all of the extra functions that complement the
tinker mass function. Specifically, it contains implementations of
the derivatives with respect to each of the tinker parameters, as
well as calculating the covariance matrix between mass bins.
"""
import numpy as np

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
