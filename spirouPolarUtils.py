# -*- coding: iso-8859-1 -*-
"""
    Created on May 7 2020
    
    Description: Library containing several utilities for the analysis of polarimetry data
    
    @author: Eder Martioli <martioli@iap.fr>
    
    Institut d'Astrophysique de Paris, France.

    """

__version__ = "1.0"

__copyright__ = """
    Copyright (c) ...  All rights reserved.
    """

from optparse import OptionParser
import os,sys

import numpy as np
import glob

import matplotlib
import matplotlib.pyplot as plt
import astropy.io.fits as fits
from scipy import stats
from scipy.optimize import curve_fit,fsolve
from scipy import constants
from scipy.integrate import simps
from scipy.stats import chisquare
from uncertainties import ufloat
from astropy.modeling.models import Voigt1D

import spirouPolar, spirouLSD

def save_rv_time_series(output, bjd, rv, rverr, time_in_rjd=True, rv_in_mps=False) :

    outfile = open(output,"w+")
    outfile.write("rjd    vrad    svrad\n")
    outfile.write("---    ----    -----\n")

    for i in range(len(bjd)) :
        
        if time_in_rjd :
            rjd = bjd[i] - 2400000.
        else :
            rjd = bjd[i]
        
        if rv_in_mps :
            outfile.write("{0:.10f} {1:.2f} {2:.2f}\n".format(rjd, 1000. * rv[i], 1000. * rverr[i]))
        else :
            outfile.write("{0:.10f} {1:.5f} {2:.5f}\n".format(rjd, rv[i], rverr[i]))

    outfile.close()


def plot_2d(x, y, z, LIM=None, LAB=None, z_lim=None, title="", pfilename="", cmap="gist_heat"):
    """
    Use pcolor to display sequence of spectra
    
    Inputs:
    - x:        x array of the 2D map (if x is 1D vector, then meshgrid; else: creation of Y)
    - y:        y 1D vector of the map
    - z:        2D array (sequence of spectra; shape: (len(x),len(y)))
    - LIM:      list containing: [[lim_inf(x),lim_sup(x)],[lim_inf(y),lim_sup(y)],[lim_inf(z),lim_sup(z)]]
    - LAB:      list containing: [label(x),label(y),label(z)] - label(z) -> colorbar
    - title:    title of the map
    - **kwargs: **kwargs of the matplolib function pcolor
    
    Outputs:
    - Display 2D map of the sequence of spectra z
    
    """
    
    if len(np.shape(x))==1:
        X,Y  = np.meshgrid(x,y)
    else:
        X = x
        Y = []
        for n in range(len(x)):
            Y.append(y[n] * np.ones(len(x[n])))
        Y = np.array(Y,dtype=float)
    Z = z

    if LIM == None :
        x_lim = [np.min(X),np.max(X)] #Limits of x axis
        y_lim = [np.min(Y),np.max(Y)] #Limits of y axis
        if z_lim == None :
            z_lim = [np.min(Z),np.max(Z)]
        LIM   = [x_lim,y_lim,z_lim]

    if LAB == None :
        ### Labels of the map
        x_lab = r"$Velocity$ [km/s]"     #Wavelength axis
        y_lab = r"Time [BJD]"         #Time axis
        z_lab = r"CCF"     #Intensity (exposures)
        LAB   = [x_lab,y_lab,z_lab]

    fig = plt.figure()
    plt.rcParams["figure.figsize"] = (10,7)
    ax = plt.subplot(111)

    cc = ax.pcolor(X, Y, Z, vmin=LIM[2][0], vmax=LIM[2][1], cmap=cmap)
    cb = plt.colorbar(cc,ax=ax)
    
    ax.set_xlim(LIM[0][0],LIM[0][1])
    ax.set_ylim(LIM[1][0],LIM[1][1])
    
    ax.set_xlabel(LAB[0])
    ax.set_ylabel(LAB[1],labelpad=15)
    cb.set_label(LAB[2],rotation=270,labelpad=30)

    ax.set_title(title,pad=35)

    if pfilename=="" :
        plt.show()
    else :
        plt.savefig(pfilename, format='png')
    plt.clf()
    plt.close()

#################################################################################################
def subtract_median(ccf, vels=[], sig_clip = 0.0, ind_ini=0, ind_end=0, fit=False, fit_scale=False, verbose=False, median=True, subtract=False):
    """
        Compute the median ccf along the time axis
        Divide each exposure by the median
        
        Inputs:
        - ccf: 2D matrix (N_exposures,N_velocities) from which median is computed
        - ind_ini, ind_end: if both == 0 or ind_end<=ind_ini: compute median on all spectra
          else: ind_ini,ind_end stand for the beginning and the end of the planetary transit respectively
                then median computed on the out-of-transit spectra only
        - fit: boolean to fit median spectrum to each observation before normalizing it
        Outputs:
        - loc: python dict containing all products
    """
    
    loc = {}

    if (ind_ini == 0 and ind_end == 0) or (ind_end <= ind_ini):
        # Compute median on all spectra along the time axis
        if median :
            ccf_med = np.median(ccf,axis=0)
        else :
            ccf_med = np.nanmean(ccf,axis=0)
    else:
        # Compute median on out-of-transit spectra only
        ccf_out = np.concatenate((ccf[:ind_ini],ccf[ind_end:]),axis=0)
        if median :
            ccf_med = np.median(ccf_out,axis=0)
        else :
            ccf_med = np.nanmean(ccf_out,axis=0)

    if fit :
        shift_arr = []
        ccf_calib = []
        ccf_fit = []
        
        if fit_scale :
            scale_arr = []
            def ccf_model (vels, shift, scale):
                outmodel = scale * (ccf_med - 1.0) + 1.0 + shift
                return outmodel
        else :
            def ccf_model (vels, shift):
                outmodel = ccf_med + shift
                return outmodel

        for i in range(len(ccf)):
            
            if fit_scale :
                guess = [0.0001, 1.001]
            else :
                guess = [0.0001]

            mask = ~np.isnan(ccf[i])
            
            if len(ccf[i][mask]) > 0 :
                pfit, pcov = curve_fit(ccf_model, vels[mask], ccf[i][mask], p0=guess)
            else :
                if fit_scale :
                    pfit = [0.,1.]
                else :
                    pfit = [0.]
        
            ccf_med_fit = ccf_model(vels, *pfit)

            shift_arr.append(pfit[0])
            if fit_scale :
                scale_arr.append(pfit[1])

            ccf_fit.append(ccf_med_fit)
            if fit_scale :
                ccf_calib_loc = (ccf[i] - pfit[0] - 1.0) / pfit[1] + 1.0
            else :
                ccf_calib_loc = ccf[i] - pfit[0]

            ccf_calib.append(ccf_calib_loc)

        loc["shift"] = np.array(shift_arr, dtype=float)
        if fit_scale :
            loc["scale"] = np.array(scale_arr, dtype=float)
        ccf_calib = np.array(ccf_calib, dtype=float)
        ccf_fit = np.array(ccf_fit, dtype=float)

        if (ind_ini == 0 and ind_end == 0) or (ind_end <= ind_ini):
            # Compute median on all spectra along the time axis
            if median :
                ccf_med_new = np.median(ccf_calib,axis=0)
            else :
                ccf_med_new = np.nanmean(ccf_calib,axis=0)
        else:
            # Compute median on out-of-transit spectra only
            ccf_out_new = np.concatenate((ccf_calib[:ind_ini],ccf_calib[ind_end:]),axis=0)
            if median :
                ccf_med_new = np.median(ccf_out_new,axis=0)
            else :
                ccf_med_new = np.nanmean(ccf_out_new,axis=0)

        ccf_med = ccf_med_new
        if subtract :
            ccf_sub = ccf_calib - ccf_med + 1.0
        else :
            ccf_sub = ccf_calib / ccf_med

        residuals = ccf_calib - ccf_med
        ccf_medsig = np.median(np.abs(residuals),axis=0) / 0.67449

    else :
        # Divide or subtract each ccf by ccf_med
        if subtract :
            ccf_sub = ccf - ccf_med + 1.0
        else :
            ccf_sub = ccf / ccf_med

        residuals = ccf - ccf_med
        ccf_medsig = np.median(np.abs(residuals),axis=0) / 0.67449

    loc["ccf_med"] = ccf_med
    loc["ccf_sig"] = ccf_medsig
    loc["vels"] = vels
    loc["ccf_sub"] = ccf_sub
    loc["residuals"] = residuals

    loc["ccf"] = loc["residuals"] + loc["ccf_med"]
    
    return loc


def detrend_airmass(I_norm, vels, airmass, snr=None, deg=2, log=False, plot=False):
    
    """
        Detrend normalized CCF with airmass
        Goal: remove residuals of tellurics not subracted when dividing by median spectrum
        Use least square estimator (LSE) to estimate the components of a linear (or log) model of airmass
        
        Inputs:
        - I_norm:  2D matrix of normalised spectra (N_exposures,len(W))
        - vels:    1D velocity vector
        - airmass: 1D airmass vector
        - deg:    degree of the linear model (e.g. I(t) = a0 + a1*airmass + ... + an*airmass^(n))
        - log:    if 1 fit log(I_norm) instead of I(t)
        - plot:   if true, plot the components removed with this model
        
        Outputs:
        - I_m_tot: Sequence of spectra without the airmass detrended component
    """
    if log:
        I_tmp = np.log(I_norm)
    else:
        I_tmp = I_norm - 1.

    ### Covariance matrix of the noise from DRS SNRs
    COV_inv = np.diag(snr**(2))
    
    ### Apply least-square estimator
    X = []
    X.append(np.ones(len(I_tmp)))
    for k in range(deg):
        X.append(airmass ** (k+1))
    X = np.array(X,dtype=float).T
    A = np.dot(X.T,np.dot(COV_inv,X))
    b = np.dot(X.T,np.dot(COV_inv,I_tmp))
    #A = np.dot(X.T,X)
    #b = np.dot(X.T,I_tmp)

    I_best   = np.dot(np.linalg.inv(A),b)

    ### Plot each component estimated with LSE
    if plot:
        fig = plt.figure()
        ax  = plt.subplot(111)
        c   = 0
        col = ["red","green","magenta","cyan","blue","black","yellow"]
        for ii in I_best:
            lab = "Order " + str(c)
            alp = 1 - c/len(I_best)
            plt.plot(vels,ii,label=lab,color=col[c],zorder=c+1,alpha=alp)
            c += 1
            if c == 7: break
        plt.legend()
        plt.title("Components removed - airmass detrending")
        plt.xlabel(r"$\Velocity$ [km/s]")
        plt.ylabel("Residuals removed")
        plt.show()

    if log:
        I_m_tot  = I_tmp - np.dot(X,I_best)
        return np.exp(I_m_tot)
    else:
        I_m_tot  = I_tmp + 1 - np.dot(X,I_best)
        return I_m_tot
    #######################################################################


def sigma_clip(ccfs, nsig=3.0, plot=False, interpolate=False, replace_by_model = True) :
    
    out_ccf = np.full_like(ccfs["ccf"], np.nan)
    out_ccf_sub = np.full_like(ccfs["ccf_sub"], np.nan)

    for i in range(len(ccfs["ccf"])) :
        sigclipmask = np.abs(ccfs["residuals"][i]) > (nsig * ccfs["ccf_sig"])
        if plot :
            plt.plot(ccfs["vels"], ccfs["residuals"][i], alpha=0.3)
            if len(ccfs["residuals"][i][sigclipmask]) :
                plt.plot(ccfs["vels"][sigclipmask], ccfs["residuals"][i][sigclipmask], "ro")
    
        # set good values first
        out_ccf[i][~sigclipmask] = ccfs["ccf"][i][~sigclipmask]
        out_ccf_sub[i][~sigclipmask] = ccfs["ccf_sub"][i][~sigclipmask]
    
        # now decide what to do with outliers
        if interpolate :
            if i > 0 and i < len(ccfs["ccf"]) - 1 :
                out_ccf[i][sigclipmask] = (ccfs["ccf"][i-1][sigclipmask] + ccfs["ccf"][i+1][sigclipmask]) / 2.
                out_ccf_sub[i][sigclipmask] = (ccfs["ccf_sub"][i-1][sigclipmask] + ccfs["ccf_sub"][i+1][sigclipmask]) / 2.
            elif i == 0 :
                out_ccf[i][sigclipmask] = ccfs["ccf"][i+1][sigclipmask]
                out_ccf_sub[i][sigclipmask] = ccfs["ccf_sub"][i+1][sigclipmask]
            elif i == len(ccfs["ccf"]) - 1 :
                out_ccf[i][sigclipmask] = ccfs["ccf"][i-1][sigclipmask]
                out_ccf_sub[i][sigclipmask] = ccfs["ccf_sub"][i-1][sigclipmask]
        
        if replace_by_model :
            out_ccf[i][sigclipmask] = ccfs["ccf_med"][sigclipmask]
            out_ccf_sub[i][sigclipmask] = 1.0

        #if plot :
        #    plt.plot(ccfs["vels"][sigclipmask],out_ccf[i][sigclipmask],'b.')

    if plot :
        plt.plot(ccfs["vels"], nsig * ccfs["ccf_sig"], 'r--', lw=2)
        plt.plot(ccfs["vels"], -nsig * ccfs["ccf_sig"], 'r--', lw=2)
        plt.show()
    
    ccfs["ccf"] = out_ccf
    ccfs["ccf_sub"] = out_ccf_sub

    return ccfs

def plot_lsd_profiles(vels, zz, zz_err, zgauss, z_p, z_p_err, z_p_model, z_np, z_np_err, vel_lim=None, stokes="V"):
    plot_name = 'polar_lsd_plot'

    font = {'size': 16}
    matplotlib.rc('font', **font)
    
    # ---------------------------------------------------------------------
    # set up fig
    fig, frames = spirouPolar.setup_figure({}, ncols=1, nrows=3, sharex=True)
    # clear the current figure
    #plt.clf()
    # ---------------------------------------------------------------------
    frame = frames[0]
    frame.errorbar(vels, zz, yerr=zz_err, fmt='.', color='red')
    frame.plot(vels, zz, '-', linewidth=0.3, color='red')
    frame.plot(vels, zgauss, '-', color='green')
    if vel_lim != None :
        frame.set_xlim(vel_lim[0],vel_lim[1])
    title = 'LSD Analysis'
    ylabel = 'Stokes I'
    xlabel = ''
    # set title and labels
    frame.set(title=title, xlabel=xlabel, ylabel=ylabel)
    #frame.grid(color='gray', linestyle='dotted', linewidth=0.4)
    # ---------------------------------------------------------------------

    # ---------------------------------------------------------------------
    frame = frames[1]
    title = ''
    frame.errorbar(vels, z_p, yerr=z_p_err, fmt='.', color='blue')
    frame.plot(vels, z_p_model, '-', linewidth=1.0, color='#1f77b4')
    if vel_lim != None :
        frame.set_xlim(vel_lim[0],vel_lim[1])
    ylabel = 'Stokes {0}'.format(stokes)
    xlabel = ''
    # set title and labels
    frame.set(title=title, xlabel=xlabel, ylabel=ylabel)
    plot_y_lims = frame.get_ylim()
    y_range = np.abs(plot_y_lims[1] - plot_y_lims[0])
    #frame.grid(color='gray', linestyle='dotted', linewidth=0.4)
    # ---------------------------------------------------------------------

    # ---------------------------------------------------------------------
    frame = frames[2]
    null_mean = np.mean(z_np)
    bottom, top = null_mean - y_range/2.0, null_mean + y_range/2.0
    frame.set_ylim(bottom, top)
    frame.errorbar(vels, z_np, yerr=z_np_err, fmt='.', color='orange')
    frame.plot(vels, z_np, '-', linewidth=0.5, color='orange')
    if vel_lim != None :
        frame.set_xlim(vel_lim[0],vel_lim[1])
    xlabel = 'velocity [km/s]'
    ylabel = 'Null'
    # set title and labels
    frame.set(title=title, xlabel=xlabel, ylabel=ylabel)
    #frame.grid(color='gray', linestyle='dotted', linewidth=0.4)
    # ---------------------------------------------------------------------

    # ---------------------------------------------------------------------
    # turn off interactive plotting
    # end plotting function properly
    spirouPolar.end_plotting({}, plot_name)



def longitudinal_b_field(vels, pol, flux, l0, geff, pol_err=[], flux_err=[], fit_continuum=False, fit_polynomial=False, npcont=10, plot=False, debug=False) :

    c_kps = (constants.c / 1000.)
    
    const = - 2.14e11 / (l0 * geff * c_kps)

    int_pol = simps(vels * pol, x=vels)
    
    if len(pol_err) :
        int_pol_err = np.sqrt(simps((vels * vels) * (pol_err * pol_err), x=vels))
        int_pol = ufloat(int_pol, int_pol_err)

    cont_sample, cont_vels = flux[:npcont], vels[:npcont]
    cont_sample = np.append(cont_sample,flux[-npcont:])
    cont_vels = np.append(cont_vels,vels[-npcont:])

    if fit_continuum :
        if fit_polynomial :
            c = np.polyfit(cont_vels, cont_sample, 1)
            p = np.poly1d(c)
            cont = p(vels)

            if plot and debug:
                # plot flux profile to check continuum
                plt.plot(vels,flux,'.')
                plt.plot(cont_vels,cont_sample,'o')
                plt.plot(vels,cont,'--')
                plt.show()
        else :
            cont = np.nanmedian(cont_sample)
        int_flux = simps((1.0 - flux/cont), x=vels)
    else :
        int_flux = simps((1.0 - flux), x=vels)

    if len(flux_err) :
        #cont_err = np.median(np.abs(flux-cont)) / 0.67449
        #int_flux_err = np.sqrt(simps((cont_err*cont_err + flux_err*flux_err), x=vels))
        int_flux_err = np.sqrt(simps(flux_err*flux_err, x=vels))
        int_flux = ufloat(int_flux, int_flux_err)

    b = const * (int_pol / int_flux)

    if len(pol_err) or len(flux_err) :
        return b.nominal_value, b.std_dev
    else :
        return b, 0.



def gauss_function(x, a, x0, sigma):
    """
        A standard 1D gaussian function (for fitting against)]=
        
        :param x: numpy array (1D), the x data points
        :param a: float, the amplitude
        :param x0: float, the mean of the gaussian
        :param sigma: float, the standard deviation (FWHM) of the gaussian
        :return gauss: numpy array (1D), size = len(x), the output gaussian
        """
    return a * np.exp(-0.5 * ((x - x0) / sigma) ** 2)


def voigt_function(x, a, x0, fwhm_L, fwhm_G):
    """
        A standard 1D Voigt function (for fitting against)]=
        
        :param x: numpy array (1D), the x data points
        :param a: float, the amplitude
        :param x0: float, the mean of the voigt function
        :param fwhm_L: float, The Lorentzian full width at half maximum
        :param fwhm_G: float, The Gaussian full width at half maximum
        :return voigt: numpy array (1D), size = len(x), the output voigt
        """
    voigt_func = Voigt1D(x_0=x0, amplitude_L=a, fwhm_L=fwhm_L, fwhm_G=fwhm_G)
    
    return voigt_func(x)

def double_gaussian_pol_function(v, a, v1, v2, sig, dc):
    expcom = np.exp(- v*v / (2*sig*sig))
    exp1 = np.exp((2*v*v1 - v1*v1) / (2*sig*sig))
    exp2 = np.exp((2*v*v2 - v2*v2) / (2*sig*sig))
    return a * expcom * (exp1 - exp2) + dc


def fit_zeeman_split(vels, pol, pol_err, guess=None, func_type="voigt", plot=False) :

    if guess == None :
        amplitude = 10. * np.abs(np.max(pol) - np.min(pol)) / 2.
        cont = np.mean(pol)
        vel1 = vels[np.argmax(pol)]
        vel2 = vels[np.argmin(pol)]
        sigma = 10.
    
    if func_type == "gaussian" :
        
        def gaussian_pol_function (v, a, v1, v2, sig, c) :
            f1 = gauss_function(v, a, v1, sig)
            f2 = gauss_function(v, a, v2, sig)
            return (f1 - f2) + c
        if guess == None :
            guess = [amplitude, vel1, vel2, sigma, cont]
        
        pfit, pcov = curve_fit(gaussian_pol_function, vels, pol, p0=guess)
        pol_model = gaussian_pol_function(vels, *pfit)
        pol1_model = gauss_function(vels, pfit[0], pfit[1], pfit[3]) + pfit[4]
        pol2_model = - gauss_function(vels, pfit[0], pfit[2], pfit[3]) + pfit[4]

    elif func_type == "voigt" :
        
        def voigt_pol_function (v, a, v1, v2, sigL, sigG, c) :
            f1 = voigt_function(v, a, v1, sigL, sigG)
            f2 = voigt_function(v, a, v2, sigL, sigG)
            return (f1 - f2) + c
        
        if guess == None :
            guess = [amplitude, vel1, vel2, sigma, sigma, cont]
        
        pfit, pcov = curve_fit(voigt_pol_function, vels, pol, p0=guess)
        pol_model = voigt_pol_function(vels, *pfit)
        pol1_model = voigt_function(vels, pfit[0], pfit[1], pfit[3], pfit[4]) + pfit[5]
        pol2_model = - voigt_function(vels, pfit[0], pfit[2], pfit[3], pfit[4]) + pfit[5]

    efit = []
    for i in range(len(pfit)):
        try:
            efit.append(np.abs(pcov[i][i])**0.5)
        except:
            efit.append( 0.00 )
    efit = np.array(efit)

    deltav = ufloat(pfit[2], efit[2]) - ufloat(pfit[1], efit[1])
    vshift = (ufloat(pfit[1], efit[1]) + ufloat(pfit[2], efit[2]) )/ 2.

    if plot :
        plt.errorbar(vels,pol,yerr=pol_err, fmt='o', label="Data")
        plt.plot(vels,pol_model,'-',label="LSD Model")
        #plt.errorbar(vels,pol-pol_model,yerr=pol_err, fmt='o', label="Residuals")

        plt.plot(vels,pol1_model,'r--', label="Red-shifted model")
        plt.plot(vels,pol2_model,'b--',label="Blue-shifted model")

        ymin, ymax = plt.ylim()[0], plt.ylim()[1]
        plt.vlines(pfit[1], ymin, ymax, colors='r', lw=0.3, linestyles="dashed")
        plt.vlines(pfit[2], ymin, ymax, colors='b', lw=0.3, linestyles="dashed")
        plt.vlines(vshift.nominal_value, ymin, ymax, lw=0.7, colors='k', linestyles="solid")
        
        plt.legend()
        plt.xlabel("Velocity [km/s]")
        plt.ylabel("Degree of polarization (Stokes V)")
        plt.show()

    loc = {}
    loc["FUNC_TYPE"] = func_type
    loc["AMP"], loc["AMP_ERR"] = pfit[0], efit[0]
    loc["V1"], loc["V1_ERR"] = pfit[1], efit[1]
    loc["V2"], loc["V2_ERR"] = pfit[2], efit[2]
    loc["DELTAV"], loc["DELTAV_ERR"] = np.abs(deltav.nominal_value), deltav.std_dev
    loc["VSHIFT"], loc["VSHIFT_ERR"] = vshift.nominal_value, vshift.std_dev
    if func_type == "gaussian" :
        loc["SIG"], loc["SIG_ERR"] = pfit[3], efit[3]
        loc["CONT"], loc["CONT_ERR"] = pfit[4], efit[4]
    if func_type == "voigt" :
        loc["SIGL"], loc["SIGL_ERR"] = pfit[3], efit[3]
        loc["SIG"], loc["SIG_ERR"] = pfit[4], efit[4]
        loc["CONT"], loc["CONT_ERR"] = pfit[5], efit[5]

    loc["MODEL"] = pol_model
    loc["CHISQR"] = np.sum( (pol - pol_model)**2 / pol_err ** 2 )

    return loc


def fit_lsd_flux_profile(vels, flux, flux_err, guess=None, func_type="voigt", plot=False) :
    
    if guess == None :
        amplitude = 10. * np.abs(np.max(flux) - np.min(flux)) / 2.
        cont = np.mean(flux)
        vel_shift = vels[np.argmin(flux)]
        sigma = 10.

    if func_type == "gaussian" :
        
        def gaussian_flux_function (v, a, v0, sig, c) :
            f = gauss_function(v, a, v0, sig)
            return f + c
        if guess == None :
            guess = [amplitude, vel_shift, sigma, cont]
        
        pfit, pcov = curve_fit(gaussian_flux_function, vels, flux, p0=guess)
        flux_model = gaussian_flux_function(vels, *pfit)

    elif func_type == "voigt" :
        
        def voigt_flux_function (v, a, v0, sigL, sigG, c) :
            f = voigt_function(v, a, v0, sigL, sigG)
            return f + c
        
        if guess == None :
            guess = [amplitude, vel_shift, sigma, sigma, cont]
        
        pfit, pcov = curve_fit(voigt_flux_function, vels, flux, p0=guess)
        flux_model = voigt_flux_function(vels, *pfit)

    efit = []
    for i in range(len(pfit)):
        try:
            efit.append(np.abs(pcov[i][i])**0.5)
        except:
            efit.append( 0.00 )
    efit = np.array(efit)
    vshift = pfit[1]

    if plot :
        plt.errorbar(vels,flux,yerr=flux_err, fmt='o', label="Data")
        plt.plot(vels,flux_model,'-',label="LSD Model")
        ymin, ymax = plt.ylim()[0], plt.ylim()[1]
        plt.vlines(vshift, ymin, ymax, lw=0.7, colors='k', linestyles="solid")
        
        plt.legend()
        plt.xlabel("Velocity [km/s]")
        plt.ylabel("Flux (Stokes I)")
        plt.show()

    loc = {}
    loc["FUNC_TYPE"] = func_type
    loc["AMP"], loc["AMP_ERR"] = pfit[0], efit[0]
    loc["VSHIFT"], loc["VSHIFT_ERR"] = pfit[1], efit[1]
    if func_type == "gaussian" :
        loc["SIG"], loc["SIG_ERR"] = pfit[2], efit[2]
        loc["CONT"], loc["CONT_ERR"] = pfit[3], efit[3]
    if func_type == "voigt" :
        loc["SIGL"], loc["SIGL_ERR"] = pfit[2], efit[2]
        loc["SIG"], loc["SIG_ERR"] = pfit[3], efit[3]
        loc["CONT"], loc["CONT_ERR"] = pfit[4], efit[4]

    loc["MODEL"] = flux_model
    loc["CHISQR"] = np.sum( (flux - flux_model)**2 / flux_err ** 2 )

    return loc



def save_lsdstack_to_fits(filename, vels, flux, fluxerr, fluxmodel, pol, polerr, polmodel,  null, nullerr, base_header, blong_vel_min=-50, blong_vel_max=50) :
    """
    Function to save output FITS image to store LSD analysis.
    
    :param filename: string, Output FITS filename
    :param
        vels: numpy array (2D), LSD analysis data
        pol: numpy array (2D), LSD analysis data
        polerr: numpy array (2D), LSD analysis data
        polmodel: numpy array (2D), LSD analysis data
        flux: numpy array (2D), LSD analysis data
        fluxerr: numpy array (2D), LSD analysis data
        fluxmodel: numpy array (2D), LSD analysis data
        null: numpy array (2D), LSD analysis data
        nullerr: numpy array (2D), LSD analysis data
    """

    primary_hdu = fits.PrimaryHDU()
    
    header = primary_hdu.header
    
    header.set('ORIGIN', "spirou-polarimetry")
    
    header.set('LSDIFITF', fluxmodel["FUNC_TYPE"], 'Function type for Stokes I fit profile')
    header.set('LSDILD', fluxmodel["AMP"], 'Line depth for Stokes I fit profile')
    header.set('LSDICONT', fluxmodel["CONT"], 'Continuum for Stokes I fit profile')
    header.set('LSDISIG', fluxmodel["SIG"], 'Sigma for Stokes I fit profile [km/s]')
    if "SIGL" in fluxmodel.keys() :
        header.set('LSDISIGL', fluxmodel["SIGL"], 'Lorentz sigma for Stokes I fit profile [km/s]')
    header.set('LSDIV0', fluxmodel["VSHIFT"], 'Vel shift for Stokes I fit profile [km/s]')
    header.set('LSDICHI2', fluxmodel["CHISQR"], 'Chi-square for Stokes I fit profile')
    flux_res = flux - fluxmodel["MODEL"]
    flux_rms = np.std(flux_res)
    header.set('LSDIRRMS', flux_rms, 'RMS of residuals for Stokes I fit profile')

    header.set('LSDVFITF', polmodel["FUNC_TYPE"], 'Function type for Stokes V fit profile')
    header.set('LSDVLD', polmodel["AMP"], 'Line depth for Stokes V fit profile')
    header.set('LSDVCONT', polmodel["CONT"], 'Continuum for Stokes V fit profile')
    header.set('LSDVSIG', polmodel["SIG"], 'Gauss sigma for Stokes V fit profile [km/s]')
    if "SIGL" in polmodel.keys() :
        header.set('LSDVSIGL', polmodel["SIGL"], 'Lorentz sigma for Stokes V fit profile [km/s]')
    header.set('LSDVV0', polmodel["VSHIFT"], 'Vel shift for Stokes V fit profile [km/s]')
    header.set('LSDVV1', polmodel["V1"], 'Red line shift for Stokes V fit profile [km/s]')
    header.set('LSDVV2', polmodel["V2"], 'Blue line shift for Stokes V fit profile [km/s]')
    header.set('LSDVSPLT', polmodel["DELTAV"], 'Zeeman split for Stokes V fit profile [km/s]')
    header.set('LSDVCHI2', polmodel["CHISQR"], 'Chi-square for Stokes V fit profile')
    pol_res = pol - polmodel["MODEL"]
    pol_rms = np.std(pol_res)
    header.set('LSDVRRMS', pol_rms, 'RMS of residuals for Stokes V fit profile')

    null_rms = np.std(null)
    header.set('NULLRMS', null_rms, 'RMS of null polarization profile')

    header.set('MASKFILE', base_header["MASKFILE"], 'Mask file used in LSD analysis')
    header.set('NLINMASK', base_header["MASKFILE"], 'Number of lines in the original mask')
    header.set('NLINUSED', base_header["NLINUSED"], 'Number of lines used in LSD analysis')
    header.set('WAVEAVG', base_header["WAVEAVG"], 'Mean wavelength of lines used in LSD analysis')
    header.set('LANDEAVG', base_header["LANDEAVG"], 'Mean lande of lines used in LSD analysis')

    mask = vels > blong_vel_min
    mask &= vels < blong_vel_max

    b_field, b_field_err = longitudinal_b_field(vels[mask], pol[mask], flux[mask], base_header["WAVEAVG"], base_header["LANDEAVG"], pol_err=polerr[mask], flux_err=fluxerr[mask], npcont=2, plot=True)
    print("B_l = {0:.2f} +- {1:.2f} Gauss".format(b_field, b_field_err))

    header.set('BLONG', b_field, 'Longitudinal magnetic field (G)')
    header.set('BLONGERR', b_field_err, 'Longitudinal magnetic field error (G)')

    hdu_vels = fits.ImageHDU(data=vels, name="Velocity")
    hdu_pol = fits.ImageHDU(data=pol, name="StokesVQU")
    hdu_pol_err = fits.ImageHDU(data=polerr, name="StokesVQU_Err")
    hdu_polmodel = fits.ImageHDU(data=polmodel["MODEL"], name="StokesVQUModel")
    hdu_flux = fits.ImageHDU(data=flux, name="StokesI")
    hdu_flux_err = fits.ImageHDU(data=fluxerr, name="StokesI_Err")
    hdu_fluxmodel = fits.ImageHDU(data=fluxmodel["MODEL"], name="StokesIModel")
    hdu_null = fits.ImageHDU(data=null, name="Null")
    hdu_null_err = fits.ImageHDU(data=nullerr, name="Null_Err")

    mef_hdu = fits.HDUList([primary_hdu, hdu_vels, hdu_pol, hdu_pol_err, hdu_polmodel, hdu_flux, hdu_flux_err, hdu_fluxmodel, hdu_null, hdu_null_err])

    mef_hdu.writeto(filename, overwrite=True)
