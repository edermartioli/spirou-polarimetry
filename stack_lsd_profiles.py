# -*- coding: iso-8859-1 -*-
"""
    Created on May 7 2020
    
    Description: This routine stacks a series of LSD profiles
    
    @author: Eder Martioli <martioli@iap.fr>
    
    Institut d'Astrophysique de Paris, France.
    
    Simple usage examples:
    
    python stack_lsd_profiles.py --input=*_lsd.fits

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
import astropy.io.fits as fits

from scipy.interpolate import interp1d
from scipy import ndimage

import spirouPolarUtils as spu


def combine_profiles(vels, array, err, median=True, plot=False) :
    
    mean_pol = np.zeros_like(vels)
    sigma_pol = np.zeros_like(vels)
    
    for i in range(len(vels)) :
        if median :
            mean_pol[i] = np.nanmedian(array[:,i])
        else :
            mean_pol[i] = np.nanmean(array[:,i])

        sigma_pol[i] = np.sqrt(np.nansum(err[:,i] * err[:,i])) / float(len(err[:,i]))

    if plot :
        plt.errorbar(vels, mean_pol, yerr=sigma_pol, fmt='o')
        plt.show()

    return mean_pol, sigma_pol


parser = OptionParser()
parser.add_option("-i", "--input", dest="input", help="Input LSD data pattern",type='string',default="*_lsd.fits")
parser.add_option("-o", "--output", dest="output", help="Output stack LSD FITS file",type='string',default="")
parser.add_option("-r", "--source_rv", dest="source_rv", help="Source radial velocity in km/s",type='float',default=0.)
parser.add_option("-t", action="store_true", dest="timecombine", help="timecombine", default=False)
parser.add_option("-p", action="store_true", dest="plot", help="plot", default=False)
parser.add_option("-v", action="store_true", dest="verbose", help="verbose", default=False)

try:
    options,args = parser.parse_args(sys.argv[1:])
except:
    print("Error: check usage with stack_lsd_profiles.py -h ")
    sys.exit(1)

if options.verbose:
    print('Input LSD data pattern: ', options.input)
    print('Output stack LSD FITS file: ', options.output)

# make list of data files
if options.verbose:
    print("Creating list of lsd files...")
inputdata = sorted(glob.glob(options.input))
#---

source_rv = options.source_rv

bjd = []
lsd_vels = []
lsd_pol, lsd_null, lsd_flux  = [], [], []
lsd_pol_err, lsd_null_err, lsd_flux_err  = [], [], []
lsd_pol_corr, lsd_pol_model_corr  = [], []
lsd_flux_corr  = []

pol_rv, zeeman_split = [], []
pol_line_depth, pol_fwhm = [], []
shift = 0.
vel_min, vel_max = -50, 50

for i in range(len(inputdata)) :
    print("Loading LSD profile in file {0}/{1}: {2}".format(i, len(inputdata), inputdata[i]))
    hdu = fits.open(inputdata[i])
    hdr = hdu[0].header + hdu[1].header

    try :
        #stokesI_fit = spu.fit_lsd_flux_profile(hdu['VELOCITY'].data, hdu['STOKESI'].data, hdu['STOKESI_ERR'].data, guess=None, func_type="voigt", plot=False)
        stokesI_fit = spu.fit_lsd_flux_profile(hdu['VELOCITY'].data, hdu['STOKESI'].data, hdu['STOKESI_ERR'].data, guess=None, func_type="gaussian", plot=False)
    except :
        print("WARNING: Could not fit gaussian to Stokes I profile, skipping file {0}: {2}".format(i, inputdata[i]))
        continue
    
    if "MEANBJD" in hdr.keys() :
        bjd.append(float(hdr["MEANBJD"]))
    elif "BJD" in hdr.keys() :
        bjd.append(float(hdr["BJD"]))
    else :
        print("Could not read BJD from header, exit ...")
        exit()

    if i == 0 :
        base_header = hdr
        vels = hdu['VELOCITY'].data
        mask = vels > vel_min
        mask &= vels < vel_max

    lsd_vels.append(hdu['VELOCITY'].data)
    lsd_pol.append(hdu['STOKESVQU'].data)
    lsd_null.append(hdu['NULL'].data)
    lsd_flux.append(hdu['STOKESI'].data)

    lsd_pol_err.append(hdu['STOKESVQU_ERR'].data)
    lsd_null_err.append(hdu['NULL_ERR'].data)
    lsd_flux_err.append(hdu['STOKESI_ERR'].data)

    if source_rv == 0. :
        vels_corr = hdu['VELOCITY'].data - stokesI_fit["VSHIFT"]
        pol_rv.append(stokesI_fit["VSHIFT"])
    else :
        vels_corr = hdu['VELOCITY'].data - source_rv
        pol_rv.append(source_rv)

    interp_pol_corr = interp1d(vels_corr, hdu['STOKESVQU'].data, kind='cubic')
    interp_flux_corr = interp1d(vels_corr, hdu['STOKESI'].data, kind='cubic')

    lsd_flux_corr.append(interp_flux_corr(vels[mask]))
    lsd_pol_corr.append(interp_pol_corr(vels[mask]))


bjd = np.array(bjd)
lsd_vels = np.array(lsd_vels, dtype=float)

lsd_pol = np.array(lsd_pol, dtype=float)
lsd_pol_err = np.array(lsd_pol_err, dtype=float)
lsd_pol_corr = np.array(lsd_pol_corr, dtype=float)

lsd_flux = np.array(lsd_flux, dtype=float)
lsd_flux_err = np.array(lsd_flux_err, dtype=float)
lsd_flux_corr = np.array(lsd_flux_corr, dtype=float)

lsd_null = np.array(lsd_null, dtype=float)
lsd_null_err = np.array(lsd_null_err, dtype=float)

pol_rv = np.array(pol_rv)

"""
plt.plot(bjd, pol_rv, '-', label="Polarimetry RV")
plt.ylabel("radial velocity [km/s]")
plt.xlabel("BJD")
plt.legend()
plt.show()
"""

ind_ini, ind_end = 0, 0

vels -= source_rv
vels = vels[mask]

lsd_pol = lsd_pol[:,mask]
lsd_pol_err = lsd_pol_err[:,mask]

median_flux = np.median(lsd_flux[:,mask])
lsd_flux = lsd_flux[:,mask] / median_flux
lsd_flux_err = lsd_flux_err[:,mask] / median_flux

lsd_null = lsd_null[:,mask] - np.median(lsd_null[:,mask])
lsd_null_err = lsd_null_err[:,mask]

# set 2D plot parameters
if options.plot :
    x_lab = r"$Velocity$ [km/s]"     #Wavelength axis
    y_lab = r"Time [BJD]"         #Time axis
    z_lab_pol = r"Degree of polarization (Stokes V)"     #Intensity (exposures)
    z_lab_null = r"Null polarization (Stokes V)"     #Intensity (exposures)
    z_lab_flux = r"Intensity (Stokes I)"     #Intensity (exposures)
    #color_map = plt.cm.get_cmap('coolwarm')
    color_map = plt.cm.get_cmap('seismic')
    reversed_color_map = color_map.reversed()
    LAB_pol  = [x_lab,y_lab,z_lab_pol]
    LAB_null  = [x_lab,y_lab,z_lab_null]
    LAB_flux  = [x_lab,y_lab,z_lab_flux]

# Stokes I (flux) LSD profiles:
reduced_lsd_flux = spu.subtract_median(lsd_flux_corr, vels=vels, ind_ini=ind_ini, ind_end=ind_end, fit=True, verbose=False, median=True, subtract=False)
reduced_lsd_flux = spu.subtract_median(reduced_lsd_flux['ccf'], vels=vels, ind_ini=ind_ini, ind_end=ind_end, fit=True, verbose=False, median=True, subtract=False)
reduced_lsd_flux = spu.subtract_median(reduced_lsd_flux['ccf'], vels=vels, ind_ini=ind_ini, ind_end=ind_end, fit=True, verbose=False, median=True, subtract=False)

if options.plot :
    spu.plot_2d(reduced_lsd_flux['vels'], bjd, reduced_lsd_flux['ccf'], LAB=LAB_flux, title="LSD Stokes I profiles", cmap=reversed_color_map)
#--------------------------

# Polarimetry LSD Stokes V profiles -- RV corrected using the RV obtained from voigt model to the zeeman split:
#reduced_lsd_pol_corr = spu.subtract_median(lsd_pol_corr, vels=vels, ind_ini=ind_ini, ind_end=ind_end, fit=True, verbose=False, median=False, subtract=True)
reduced_lsd_pol_corr = spu.subtract_median(lsd_pol_corr, vels=vels, ind_ini=ind_ini, ind_end=ind_end, fit=True, verbose=False, median=True, subtract=True)
reduced_lsd_pol_corr = spu.subtract_median(reduced_lsd_pol_corr['ccf'], vels=vels, ind_ini=ind_ini, ind_end=ind_end, fit=True, verbose=False, median=True, subtract=True)
reduced_lsd_pol_corr = spu.subtract_median(reduced_lsd_pol_corr['ccf'], vels=vels, ind_ini=ind_ini, ind_end=ind_end, fit=True, verbose=False, median=True, subtract=True)

if options.plot :
    spu.plot_2d(reduced_lsd_pol_corr['vels'], bjd, reduced_lsd_pol_corr['ccf'], LAB=LAB_pol, title="LSD Stokes V profiles", cmap=reversed_color_map)
    spu.plot_2d(reduced_lsd_pol_corr['vels'], bjd, (reduced_lsd_pol_corr['ccf_sub'] - 1.0), LAB=LAB_pol, title="Median-subtracted LSD Stokes V profiles", cmap=reversed_color_map)
    

#lsd_pol_corr_res = ndimage.median_filter(reduced_lsd_pol_corr['ccf_sub'], size=3)
#if options.plot :
#    spu.plot_2d(reduced_lsd_pol_corr['vels'], bjd, lsd_pol_corr_res, LAB=LAB_pol, title="Median-subtracted airmass detrended smoothed LSD Stokes V profiles", cmap=reversed_color_map)
#--------------------------

# Polarimetry LSD Null profiles:
reduced_lsd_null = spu.subtract_median(lsd_null, vels=vels, ind_ini=ind_ini, ind_end=ind_end, fit=True, verbose=False, median=False, subtract=True)
if options.plot :
    spu.plot_2d(reduced_lsd_null['vels'], bjd, reduced_lsd_null['ccf'], LAB=LAB_null, title="LSD null profiles", cmap=reversed_color_map)
#--------------------------

#----- plot mean profiles
vels = reduced_lsd_pol_corr["vels"]

if options.timecombine :
    # Use median profile and errors from time average
    z_p, z_p_err = reduced_lsd_pol_corr["ccf_med"], reduced_lsd_pol_corr["ccf_sig"]
    z_np, z_np_err = reduced_lsd_null["ccf_med"], reduced_lsd_null["ccf_sig"]
else :
    # Calculate median profiles and propagate statistical errors
    z_p, z_p_err = combine_profiles(vels, reduced_lsd_pol_corr['ccf'], lsd_pol_err)
    z_np, z_np_err = combine_profiles(vels, reduced_lsd_null['ccf'], lsd_null_err)

# Fit model to mean LSD polarization profile:
zeeman = spu.fit_zeeman_split(reduced_lsd_pol_corr["vels"], reduced_lsd_pol_corr["ccf_med"], pol_err=reduced_lsd_pol_corr["ccf_sig"], func_type="gaussian", plot=False)

amplitude, cont = zeeman["AMP"], zeeman["CONT"]
vel1, vel2, sigma = zeeman["V1"], zeeman["V2"], zeeman["SIG"]
guess = [amplitude, vel1, vel2, sigma, sigma, cont]

try :
    zeeman_voigt = spu.fit_zeeman_split(reduced_lsd_pol_corr["vels"], reduced_lsd_pol_corr["ccf_med"], reduced_lsd_pol_corr["ccf_sig"], guess=guess, func_type="voigt", plot=False)
except :
    zeeman_voigt = zeeman
    print("WARNING: could not fit Voigt to polar LSD profile, adopting Gaussian model")
#----------------------------------------------

if options.timecombine :
    # Use median profile and errors from time average
    zz, zz_err = reduced_lsd_flux["ccf_med"], reduced_lsd_flux["ccf_sig"]
else :
    # Calculate median flux profile and propagate statistical errors
    zz, zz_err = combine_profiles(vels, reduced_lsd_flux['ccf'], lsd_flux_err)

#----------------------------------------------
# fit gaussian or voigt function to the measured flux LSD profile
flux_model = spu.fit_lsd_flux_profile(vels, zz, zz_err, guess=None, func_type="gaussian", plot=False)

try :
    amplitude, cont = flux_model["AMP"], flux_model["CONT"]
    vel_shift, sigma = flux_model["VSHIFT"], flux_model["SIG"]
    guess = [amplitude, vel_shift, sigma, sigma, cont]

    lsd_flux_model = spu.fit_lsd_flux_profile(vels, zz, zz_err, guess=None, func_type="voigt", plot=False)
except :
    lsd_flux_model = flux_model
    print("WARNING: could not fit Voigt to Stokes I LSD profile, adopting Gaussian model")

# plot all profiles and models together
spu.plot_lsd_profiles(vels, zz, zz_err, lsd_flux_model["MODEL"], z_p, z_p_err, zeeman_voigt["MODEL"], z_np, z_np_err)
#--------------------------

# save stack LSD profiles to fits file
if options.output != "" :
    spu.save_lsdstack_to_fits(options.output, vels, zz, zz_err, lsd_flux_model, z_p, z_p_err, zeeman_voigt, z_np, z_np_err, base_header=base_header)
#--------------------------
