# -*- coding: iso-8859-1 -*-
"""
    Created on June 6 2020
    
    Description: Calculate longitudinal magnetic field (B-long) time series analysis for an input list of SPIRou LSD files
    
    @author: Eder Martioli <martioli@iap.fr>
    
    Institut d'Astrophysique de Paris, France.
    
    Simple usage examples:
    
    python spirou_blong_timeseries.py --input=*_lsd.fits
    """

__version__ = "1.0"

__copyright__ = """
    Copyright (c) ...  All rights reserved.
    """

from optparse import OptionParser
import os,sys

import numpy as np
import glob

import matplotlib.pyplot as plt
import matplotlib
import astropy.io.fits as fits

from scipy.interpolate import interp1d
from scipy import ndimage, misc

import spirouPolarUtils as spu

import spirouLSD


def measure_rvs(inputdata, nsigclip=5, set_all_rvs_to_systemic=True, sysrv_type=1, vel_min=-1e50, vel_max=+1e50, plot=False, verbose=False) :
    
    lsd_flux = []
    rv = np.array([])
    for i in range(len(inputdata)) :
        hdu = fits.open(inputdata[i])
        hdr = hdu[0].header + hdu[1].header
        if i == 0 :
            vels = hdu['VELOCITY'].data
        lsd_flux.append(hdu['STOKESI'].data)
        try :
            stokesI_fit = spu.fit_lsd_flux_profile(hdu['VELOCITY'].data, hdu['STOKESI'].data, hdu['STOKESI_ERR'].data, guess=None, func_type="gaussian", plot=False)
            rv = np.append(rv, stokesI_fit["VSHIFT"])
        except :
            rv = np.append(rv, np.nan)
            continue

    systemic_rv1 = np.nanmedian(rv)
    
    lsd_flux = np.array(lsd_flux, dtype=float)

    lsd_template = spu.subtract_median(lsd_flux, vels=vels, fit=True, verbose=False, median=True, subtract=True)

    min = np.argmin(lsd_template['ccf_med'])

    rv_min = vels[min]

    fitrange = vels - rv_min > vel_min
    fitrange &= vels - rv_min < vel_max

    try :
        median_stokesI_fit = spu.fit_lsd_flux_profile(vels[fitrange], lsd_template['ccf_med'][fitrange], lsd_template['ccf_sig'][fitrange], guess=None, func_type="gaussian", plot=plot)
        systemic_rv2 = median_stokesI_fit["VSHIFT"]
    except :
        systemic_rv2 = systemic_rv1

    sigma = np.nanmean(lsd_template["ccf_sig"])

    for i in range(len(inputdata)) :
        
        loc_sigma = np.nanstd(lsd_template["residuals"][i])
        
        if verbose :
            print("Exposure {0}/{1} SysRV={2:.3f} km/s RV={3:.3f} km/s rms={4:.1f} x sigma".format(i, len(inputdata), systemic_rv1, rv[i], loc_sigma/sigma))
        
        if loc_sigma/sigma < nsigclip :
            try :
                stokesI_fit = spu.fit_lsd_flux_profile(vels[fitrange], lsd_template['ccf'][i][fitrange], lsd_template['ccf_sig'][fitrange], guess=None, func_type="gaussian", plot=False)
                rv[i] = stokesI_fit["VSHIFT"]
            except :
                rv[i] = systemic_rv2
        else :
            rv[i] = np.nan
        #print(inputdata[i], i, rv[i])

    systemic_rv1 = np.nanmedian(rv)

    if set_all_rvs_to_systemic and sysrv_type == 1 :
        
        rv = np.full_like(rv,systemic_rv1)

    elif set_all_rvs_to_systemic and sysrv_type == 2 :
        
        fitrange = vels - systemic_rv1 > vel_min
        fitrange &= vels - systemic_rv1 < vel_max
        try :
            median_stokesI_fit = spu.fit_lsd_flux_profile(vels[fitrange], lsd_template['ccf_med'][fitrange], lsd_template['ccf_sig'][fitrange], guess=None, func_type="gaussian", plot=plot)
            systemic_rv2 = median_stokesI_fit["VSHIFT"]
        except :
            systemic_rv2 = systemic_rv1

        rv = np.full_like(rv,systemic_rv2)

    return rv


def load_lsd_time_series(inputdata, constant_rv=False, nsigclip=5, fit_profile=False, vel_min=-1e50, vel_max=+1e50, verbose=False) :
    
    loc = {}

    lsd_rv = measure_rvs(inputdata, nsigclip=nsigclip, set_all_rvs_to_systemic=constant_rv, vel_min=-20, vel_max=20)
    
    maxrv, minrv = np.nanmax(lsd_rv), np.nanmin(lsd_rv)
    
    bjd = []
    airmass, snr = [], []
    waveavg, landeavg = [], []
    
    lsd_pol, lsd_null, lsd_flux  = [], [], []
    lsd_pol_err, lsd_flux_err  = [], []
    lsd_pol_corr, lsd_flux_corr  = [], []

    lsd_pol_gaussmodel, lsd_pol_voigtmodel = [], []
    bfield, bfield_err = [], []
    pol_rv, zeeman_split = [], []
    pol_line_depth, pol_fwhm = [], []
    
    for i in range(len(inputdata)) :
        
        if np.isnan(lsd_rv[i]) :
            if verbose:
                print("Rejecting LSD profile in file {0}/{1}: {2}".format(i, len(inputdata), inputdata[i]))
            continue
        
        if verbose:
            print("Loading LSD profile in file {0}/{1}: {2}".format(i, len(inputdata), inputdata[i]))
        
        hdu = fits.open(inputdata[i])
        hdr = hdu[0].header + hdu[1].header

        if "MEANBJD" in hdr.keys() :
            bjd.append(float(hdr["MEANBJD"]))
        elif "BJD" in hdr.keys() :
            bjd.append(float(hdr["BJD"]))
        else :
            print("Could not read BJD from header, exit ...")
            exit()

        if "SNR33" in hdr.keys() :
            snr.append(float(hdr["SNR33"]))
        else :
            snr.append(1.0)

        airmass.append(float(hdr["AIRMASS"]))

        if i == 0 :
            vels_sup_lim = np.nanmax(hdu['VELOCITY'].data)
            vels_inf_lim = np.nanmin(hdu['VELOCITY'].data)

            if (vel_min + minrv) < vels_inf_lim :
                print("WARNING: requested RVs outside range, reseting vel_min to {:.1f} km/s".format(vels_inf_lim - minrv))
                vel_min = vels_inf_lim
            
            if (vel_max + maxrv) > vels_sup_lim :
                print("WARNING: requested RVs outside range, reseting vel_max to {:.1f} km/s".format(vels_sup_lim - maxrv))
                vel_max = vels_sup_lim

            mask = hdu['VELOCITY'].data > vel_min
            mask &= hdu['VELOCITY'].data < vel_max
            vels = hdu['VELOCITY'].data[mask]

        lsd_pol.append(hdu['STOKESVQU'].data[mask])
        lsd_null.append(hdu['NULL'].data[mask])
        lsd_flux.append(hdu['STOKESI'].data[mask])

        lsd_pol_err.append(hdu['STOKESVQU_ERR'].data[mask])
        lsd_flux_err.append(hdu['STOKESI_ERR'].data[mask])

        if fit_profile :
            # fit gaussian to the measured Stokes VQU LSD profile
            zeeman_gauss = spu.fit_zeeman_split(hdu['VELOCITY'].data[mask], hdu['STOKESVQU'].data[mask], pol_err=hdu['STOKESVQU_ERR'].data[mask], func_type="gaussian", plot=False)
            try :
                amplitude = zeeman_gauss["AMP"]
                cont = zeeman_gauss["CONT"]
                vel1 = zeeman_gauss["V1"]
                vel2 = zeeman_gauss["V2"]
                sigma = zeeman_gauss["SIG"]
                guess = [amplitude, vel1, vel2, sigma, sigma, cont]

                zeeman_voigt = spu.fit_zeeman_split(hdu['VELOCITY'].data[mask], hdu['STOKESVQU'].data[mask], pol_err=hdu['STOKESVQU_ERR'].data[mask], guess=guess, func_type="voigt", plot=False)
            except :
                try :
                    zeeman_voigt = spu.fit_zeeman_split(hdu['VELOCITY'].data[mask], hdu['STOKESVQU'].data[mask], pol_err=hdu['STOKESVQU_ERR'].data[mask], func_type="voigt", plot=False)
                except :
                    print("WARNING: could not fit voigt function")
            lsd_pol_gaussmodel.append(zeeman_gauss["MODEL"])
            lsd_pol_voigtmodel.append(zeeman_voigt["MODEL"])
            pol_rv.append(zeeman_voigt["VSHIFT"])
            zeeman_split.append(zeeman_voigt["DELTAV"])
            pol_line_depth.append(zeeman_voigt["AMP"])
            pol_fwhm.append(zeeman_voigt["SIG"])
        else :
            lsd_pol_gaussmodel.append(np.nan)
            lsd_pol_voigtmodel.append(np.nan)
            pol_rv.append(lsd_rv[i])
            zeeman_split.append(np.nan)
            pol_line_depth.append(np.nan)
            pol_fwhm.append(np.nan)

            lsd_pol_gaussmodel.append(None)
            lsd_pol_voigtmodel.append(None)

        vels_corr = hdu['VELOCITY'].data - lsd_rv[i]

        pol_fit = interp1d(vels_corr, hdu['STOKESVQU'].data, kind='cubic')
        lsd_pol_corr.append(pol_fit(hdu['VELOCITY'].data[mask]))
        flux_fit = interp1d(vels_corr, hdu['STOKESI'].data, kind='cubic')
        lsd_flux_corr.append(flux_fit(hdu['VELOCITY'].data[mask]))

        b, berr = spu.longitudinal_b_field(hdu['VELOCITY'].data[mask], hdu['STOKESVQU'].data[mask], hdu['STOKESI'].data[mask], hdr['WAVEAVG'], hdr['LANDEAVG'], pol_err=hdu['STOKESVQU_ERR'].data[mask], flux_err=hdu['STOKESI_ERR'].data[mask])

        landeavg.append(hdr['LANDEAVG'])
        waveavg.append(hdr['WAVEAVG'])

        bfield.append(b)
        bfield_err.append(berr)

        hdu.close()
            
    bjd = np.array(bjd)
    airmass, snr = np.array(airmass), np.array(snr)
    landeavg, waveavg = np.array(landeavg), np.array(waveavg)

    bfield, bfield_err = np.array(bfield), np.array(bfield_err)
    pol_rv, zeeman_split = np.array(pol_rv), np.array(zeeman_split)
    pol_line_depth, pol_fwhm = np.array(pol_line_depth), np.array(pol_fwhm)

    lsd_pol = np.array(lsd_pol, dtype=float)
    lsd_pol_err = np.array(lsd_pol_err, dtype=float)
    lsd_pol_corr = np.array(lsd_pol_corr, dtype=float)
    lsd_flux_corr = np.array(lsd_flux_corr, dtype=float)

    lsd_flux = np.array(lsd_flux, dtype=float)
    lsd_flux_err = np.array(lsd_flux_err, dtype=float)
    lsd_null = np.array(lsd_null, dtype=float)
    
    lsd_pol_gaussmodel = np.array(lsd_pol_gaussmodel, dtype=float)
    lsd_pol_voigtmodel = np.array(lsd_pol_voigtmodel, dtype=float)

    loc["SOURCE_RV"] = np.nanmedian(lsd_rv)
    loc["VELS"] = vels
    
    loc["BJD"] = bjd
    loc["AIRMASS"] = airmass
    loc["SNR"] = snr
    loc["WAVEAVG"] = waveavg
    loc["LANDEAVG"] = landeavg

    loc["LSD_RV"] = lsd_rv
    loc["POL_RV"] = pol_rv
    loc["ZEEMAN_SPLIT"] = zeeman_split
    loc["POL_LINE_DEPTH"] = pol_line_depth
    loc["POL_FWHM"] = pol_fwhm
    
    loc["BLONG"], loc["BLONG_ERR"] = bfield, bfield_err
    
    loc["LSD_POL"] = lsd_pol
    loc["LSD_FLUX"] = lsd_flux
    loc["LSD_FLUX_CORR"] = lsd_flux_corr
    loc["LSD_POL_ERR"] = lsd_pol_err
    loc["LSD_FLUX_ERR"] = lsd_flux_err
    loc["LSD_NULL"] = lsd_null
    loc["LSD_POL_CORR"] = lsd_pol_corr
    loc["LSD_POL_GAUSSMODEL"] = lsd_pol_gaussmodel
    loc["LSD_POL_VOIGTMODEL"] = lsd_pol_voigtmodel

    return loc


def calculate_blong_timeseries(lsddata, use_corr_data=True, plot=False, debug=False) :

    bjd = lsddata["BJD"]
    vels = lsddata["VELS"]
    waveavg = lsddata["WAVEAVG"]
    landeavg = lsddata["LANDEAVG"]

    blong, blong_err = [], []
    
    ## First calculate possible residual continuum from the median profile
    #  and remove it from the data.
    pol_cont = continuum_lsd_I(lsddata["VELS"], lsddata["LSD_POL_MED"], lsddata["LSD_POL_MED_ERR"],fit_continuum=False, npcont=7, plot=False)
    flux_cont = continuum_lsd_I(lsddata["VELS"], lsddata["LSD_FLUX_MED"], lsddata["LSD_FLUX_MED_ERR"],fit_continuum=False, npcont=7, plot=False)
    
    if use_corr_data :
        lsd_pol = lsddata["LSD_POL_CORR"] - pol_cont
        lsd_flux = lsddata["LSD_FLUX_CORR"] / flux_cont
    else :
        lsd_pol = lsddata["LSD_POL"] - pol_cont
        lsd_flux = lsddata["LSD_FLUX"] / flux_cont

    lsd_pol_err = lsddata["LSD_POL_ERR"]
    lsd_flux_err = lsddata["LSD_FLUX_ERR"] / flux_cont

    if debug :
        for i in range(len(bjd)) :
            plt.plot(lsddata["VELS"], lsd_flux[i], '.', alpha=0.3)
        plt.plot(lsddata["VELS"], lsddata["LSD_FLUX_MED"]/flux_cont, '-', lw=2)
        plt.plot(lsddata["VELS"], np.full_like(lsddata["VELS"], 1.), '-', lw=2)
        plt.show()

        for i in range(len(bjd)) :
            plt.plot(lsddata["VELS"], lsd_pol[i], '.', alpha=0.3)
        plt.plot(lsddata["VELS"], lsddata["LSD_POL_MED"]-pol_cont, '-', lw=2)
        plt.plot(lsddata["VELS"], np.full_like(lsddata["VELS"], 0.), '-', lw=2)
        plt.show()
    ##-------------------
    
    for i in range(len(bjd)) :
        b, berr = spu.longitudinal_b_field(vels, lsd_pol[i], lsd_flux[i], waveavg[i], landeavg[i], pol_err=lsd_pol_err[i], flux_err=lsd_flux_err[i])

        blong.append(b)
        blong_err.append(berr)
    
    blong = np.array(blong)
    blong_err = np.array(blong_err)

    lsddata["BLONG"], lsddata["BLONG_ERR"] = blong, blong_err

    bmed, bmederr = spu.longitudinal_b_field(vels, lsddata["LSD_POL_MED"] - pol_cont, lsddata["LSD_FLUX_MED"]/flux_cont, np.mean(waveavg), np.mean(landeavg), pol_err=lsddata["LSD_POL_MED_ERR"], flux_err=lsddata["LSD_FLUX_MED_ERR"]/flux_cont)

    if plot :
        font = {'size': 16}
        matplotlib.rc('font', **font)

        plt.errorbar(lsddata["BJD"], lsddata["BLONG"], yerr=lsddata["BLONG_ERR"], fmt='.', color="olive", label=r"B$_l$")
        
        plt.axhline(y=bmed-bmederr, ls='--', lw=1, color="orange")
        plt.axhline(y=bmed, ls='-', lw=2, color="blue", label=r"Mean B$_l$={0:.1f}+-{1:.1f} G".format(bmed, bmederr))
        plt.axhline(y=bmed+bmederr, ls='--', lw=1, color="orange")
        
        plt.plot(lsddata["BJD"], lsddata["BLONG"], '-', lw=0.7, color="olive")
        plt.ylabel("Longitudinal magnetic field [G]")
        plt.xlabel("BJD")
        plt.legend()
        plt.show()

    return lsddata


def save_blong_time_series(output, bjd, blong, blongerr, time_in_rjd=False) :
    
    outfile = open(output,"w+")
    
    for i in range(len(bjd)) :
        
        if time_in_rjd :
            time = bjd[i] - 2400000.
        else :
            time = bjd[i]
        
        outfile.write("{0:.10f} {1:.5f} {2:.5f}\n".format(time, blong[i], blongerr[i]))

    outfile.close()


def reduce_lsddata(lsddata, niter=3, apply_median_filter=True, median_filter_size=3, plot=False) :
    
    bjd = lsddata["BJD"]
    vels = lsddata["VELS"]

    if plot :
        x_lab = r"$Velocity$ [km/s]"     #Wavelength axis
        y_lab = r"Time [BJD]"         #Time axis
        z_lab_pol = r"Degree of polarization (Stokes V)"     #Intensity (exposures)
        z_lab_null = r"Null polarization (Stokes V)"     #Intensity (exposures)
        z_lab_flux = r"Intensity (Stokes I)"     #Intensity (exposures)
        LAB_pol  = [x_lab,y_lab,z_lab_pol]
        LAB_null  = [x_lab,y_lab,z_lab_null]
        LAB_flux  = [x_lab,y_lab,z_lab_flux]

    lsd_pol_corr = spu.subtract_median(lsddata["LSD_POL_CORR"], vels=vels, fit=True, verbose=False, median=True, subtract=True)
    lsd_flux_corr = spu.subtract_median(lsddata["LSD_FLUX_CORR"], vels=vels, fit=True, verbose=False, median=True, subtract=True)
    lsd_pol = spu.subtract_median(lsddata["LSD_POL"], vels=vels, fit=True, verbose=False, median=True, subtract=True)
    lsd_flux = spu.subtract_median(lsddata["LSD_FLUX"], vels=vels, fit=True, verbose=False, median=True, subtract=True)
    lsd_null = spu.subtract_median(lsddata["LSD_NULL"] - np.median(lsddata["LSD_NULL"]), vels=vels, fit=True, verbose=False, median=True, subtract=True)

    # Polarimetry LSD Stokes V profiles:
    for iter in range(niter) :
        lsd_pol = spu.subtract_median(lsd_pol['ccf'], vels=vels, fit=True, verbose=False, median=True, subtract=True)
        lsd_flux = spu.subtract_median(lsd_flux['ccf'], vels=vels, fit=True, verbose=False, median=True, subtract=True)
        lsd_pol_corr = spu.subtract_median(lsd_pol_corr['ccf'], vels=vels, fit=True, verbose=False, median=True, subtract=True)
        lsd_flux_corr = spu.subtract_median(lsd_flux_corr['ccf'], vels=vels, fit=True, verbose=False, median=True, subtract=True)
        lsd_null = spu.subtract_median(lsd_null['ccf'], vels=vels, fit=True, verbose=False, median=True, subtract=True)

    if plot :
        spu.plot_2d(lsd_pol_corr['vels'], bjd, lsd_pol_corr['ccf'], LAB=LAB_pol, title="LSD Stokes V profiles", cmap="seismic")
        spu.plot_2d(lsd_flux_corr['vels'], bjd, lsd_flux_corr['ccf'], LAB=LAB_flux, title="LSD Stokes I profiles", cmap="seismic")

    if apply_median_filter :
        lsd_pol_medfilt = ndimage.median_filter(lsd_pol['ccf'], size=median_filter_size)
        lsd_flux_medfilt = ndimage.median_filter(lsd_flux['ccf'], size=median_filter_size)
        lsd_pol_corr_medfilt = ndimage.median_filter(lsd_pol_corr['ccf'], size=median_filter_size)
        lsd_flux_corr_medfilt = ndimage.median_filter(lsd_flux_corr['ccf'], size=median_filter_size)
        lsd_null_medfilt = ndimage.median_filter(lsd_null['ccf'], size=median_filter_size)

        if plot :
            spu.plot_2d(lsd_pol_corr['vels'], bjd, lsd_pol_corr_medfilt, LAB=LAB_pol, title="Median-filtered LSD Stokes V profiles", cmap="seismic")
            spu.plot_2d(lsd_flux_corr['vels'], bjd, lsd_flux_corr_medfilt, LAB=LAB_flux, title="Median-filtered LSD Stokes I profiles", cmap="seismic")

        lsddata["LSD_POL"] = lsd_pol_medfilt
        lsddata["LSD_FLUX"] = lsd_flux_medfilt
        lsddata["LSD_POL_CORR"] = lsd_pol_corr_medfilt
        lsddata["LSD_FLUX_CORR"] = lsd_flux_corr_medfilt
        lsddata["LSD_NULL"] = lsd_null_medfilt

    else :
        lsddata["LSD_POL"] = lsd_pol['ccf']
        lsddata["LSD_FLUX"] = lsd_flux['ccf']
        lsddata["LSD_POL_CORR"] = lsd_pol_corr['ccf']
        lsddata["LSD_FLUX_CORR"] = lsd_flux_corr['ccf']
        lsddata["LSD_NULL"] = lsd_null['ccf']

    lsddata["LSD_POL_MED"] = lsd_pol_corr['ccf_med']
    lsddata["LSD_POL_MED_ERR"] = lsd_pol_corr['ccf_sig']
    lsddata["LSD_FLUX_MED"] = lsd_flux_corr['ccf_med']
    lsddata["LSD_FLUX_MED_ERR"] = lsd_flux_corr['ccf_sig']

    return lsddata


def continuum_lsd_I(vels, flux, fluxerr, fit_continuum=True, npcont=10, plot=False) :
    
    cont_sample, cont_vels = flux[:npcont], vels[:npcont]
    cont_sample = np.append(cont_sample,flux[-npcont:])
    cont_vels = np.append(cont_vels,vels[-npcont:])
    
    if fit_continuum :
        c = np.polyfit(cont_vels, cont_sample, 1)
        p = np.poly1d(c)
        cont = p(vels)
    else :
        c = np.nanmedian(cont_sample)
        cont = np.full_like(flux, c)

    if plot :
        # plot flux profile to check continuum
        plt.errorbar(vels, flux, fluxerr, fmt='.')
        plt.plot(cont_vels, cont_sample, 'o')
        plt.plot(vels, cont, '--')
        plt.show()

    return cont


parser = OptionParser()
parser.add_option("-i", "--input", dest="input", help="Input LSD data pattern",type='string',default="*_lsd.fits")
parser.add_option("-o", "--output", dest="output", help="Output B-long time series file",type='string',default="")
parser.add_option("-1", "--min_vel", dest="min_vel", help="Minimum velocity [km/s]",type='float',default=-35.)
parser.add_option("-2", "--max_vel", dest="max_vel", help="Maximum velocity [km/s]",type='float',default=35.)
parser.add_option("-s", "--nsigclip", dest="nsigclip", help="Threshold in number of sigmas to keep LSD",type='float',default=5.)
parser.add_option("-c", action="store_true", dest="constant_rv", help="Set all profiles with a constant velocity", default=False)
parser.add_option("-m", action="store_true", dest="median_filter", help="Apply median filter to polar time series", default=False)
parser.add_option("-p", action="store_true", dest="plot", help="plot", default=False)
parser.add_option("-v", action="store_true", dest="verbose", help="verbose", default=False)

try:
    options,args = parser.parse_args(sys.argv[1:])
except:
    print("Error: check usage with spirou_blong_timeseries.py -h ")
    sys.exit(1)

if options.verbose:
    print('Input LSD data pattern: ', options.input)
    print('Output Blong time series file: ', options.output)
    print('Minimum velocity = {0:.2f} km/s: '.format(options.min_vel))
    print('Maximum velocity = {0:.2f} km/s: '.format(options.max_vel))
    print('Threshold in number of sigmas to keep LSD: {0:.0f}'.format(options.nsigclip))

# make list of data files
if options.verbose:
    print("Creating list of lsd files...")
inputdata = sorted(glob.glob(options.input))
#---

vel_min, vel_max = options.min_vel, options.max_vel

lsddata = load_lsd_time_series(inputdata, constant_rv=options.constant_rv, nsigclip=options.nsigclip, vel_min=vel_min, vel_max=vel_max, verbose=options.verbose)

lsddata = reduce_lsddata(lsddata, apply_median_filter=options.median_filter, median_filter_size=(5,2), plot=False)

lsddata = calculate_blong_timeseries(lsddata, use_corr_data=True, plot=options.plot)

if options.output != "" :
    save_blong_time_series(options.output, lsddata["BJD"], lsddata["BLONG"], lsddata["BLONG_ERR"])
