#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Spirou Least-Squares Deconvolution (LSD) analysis module

Created on 2018-08-08 at 14:53

@author: E. Martioli

"""
import numpy as np
from scipy import constants
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
import os

from copy import copy, deepcopy
import matplotlib.pyplot as plt

import glob

import astropy.io.fits as fits

from scipy.sparse import csr_matrix

# =============================================================================
# Define variables
# =============================================================================
# Name of program
__NAME__ = 'spirouLSD.py'
# -----------------------------------------------------------------------------

# =============================================================================
# Define user functions
# =============================================================================

def lsd_analysis_wrapper(p, loc, verbose=False):
    """
        Function to call functions to perform Least Squares Deconvolution (LSD)
        analysis on the polarimetry data.
        
        :param p: parameter dictionary, ParamDict containing constants
        Must contain at least:
        LOG_OPT: string, option for logging
        
        :param loc: parameter dictionary, ParamDict containing data
        
        :return loc: parameter dictionary, the updated parameter dictionary
        Adds/updates the following:
        
        """
    # func_name = __NAME__ + '.lsd_analysis_wrapper()'
    name = 'LSDAnalysis'

    if verbose :
        # log start of LSD analysis calculations
        wmsg = 'Running function {0} to perform LSD analysis'
        print('info', wmsg.format(name))

    # load spectral lines
    loc = load_lsd_spectral_lines(p, loc, verbose)
    
    # get wavelength ranges covering spectral lines in the ccf mask
    loc = get_wl_ranges(p, loc)

    # prepare polarimetry data
    loc = prepare_polarimetry_data(p, loc)

    # call function to perform lsd analysis
    loc = lsd_analysis(p, loc)
    
    return loc


def load_lsd_spectral_lines(p, loc, verbose=False):
    """
    Function to load spectral lines data for LSD analysis.
    
    :param p: parameter dictionary, ParamDict containing constants
        Must contain at least:
            LOG_OPT: string, option for logging
            IC_POLAR_LSD_CCFLINES: list of strings, list of files containing
                                   spectral lines data
            IC_POLAR_LSD_WLRANGES: array of float pairs for wavelength ranges
            IC_POLAR_LSD_MIN_LINEDEPTH: float, line depth threshold
    :param loc: parameter dictionary, ParamDict to store data
        loc['LSD_MASK_FILE']: string, filename with CCF lines

    :return loc: parameter dictionaries,
        The updated parameter dictionary adds/updates the following:
        loc['LSD_LINES_WLC']: numpy array (1D), central wavelengths
        loc['LSD_LINES_ZNUMBER']: numpy array (1D), atomic number (Z)
        loc['LSD_LINES_DEPTH']: numpy array (1D), line depths
        loc['LSD_LINES_POL_WEIGHT']: numpy array (1D), line weights =
                                     depth * lande * wlc
    """

    func_name = __NAME__ + '.load_lsd_spectral_lines()'

    # if path exists use it
    if os.path.exists(loc['LSD_MASK_FILE']):
        if verbose :
            wmsg = 'Line mask used for LSD computation: {0}'
            print('info', wmsg.format(loc['LSD_MASK_FILE']))
        
        #Columns in the file are:
        # (0) wavelength (nm)
        # (1) a species code (atomic number + ionization/100, so 0.00 = neutral
        #        0.01 = singly ionized)
        # (2) an estimate of the line depth from continuum
        # (3) the excitation potential of the lower level (in eV)
        # (4) the effective Lande factor, and
        # (5) a flag for whether the line is used (1 = use).
        
        # load spectral lines data from file
        wlcf,znf,depthf,excpotf,landef,flagf = np.loadtxt(loc['LSD_MASK_FILE'],
                                               delimiter='  ',
                                               skiprows=1,
                                               usecols=(0,1,2,3,4,5),
                                               unpack=True)
    
    # else raise error
    else:
        if verbose :
            emsg = 'LSD Line mask file: "{0}" not found, unable to proceed'
            print('error', emsg.format(loc['LSD_MASK_FILE']))
        wlcf, znf, depthf, landef = None, None, None, None
        excpotf, flagf = None, None

    loc["NUMBER_OF_LINES_IN_MASK"] = len(wlcf)
    if verbose :
        print("Number of lines in the original mask = ", len(wlcf))

    # initialize data vectors
    wlc, zn, depth, lande = [], [], [], []
    excpot, flag = [], []

    # mask to use only lines with flag=1.
    flagmask = np.where(flagf == 1)
    # loop over spectral ranges to select only spectral lines within ranges
    for wlrange in p['IC_POLAR_LSD_WLRANGES']:
        # set initial and final wavelengths in range
        wl0, wlf = wlrange[0], wlrange[1]
        # create wavelength mask to limit wavelength range
        mask = np.where(np.logical_and(wlcf[flagmask] > wl0, wlcf[flagmask] < wlf))
        wlc = np.append(wlc, wlcf[flagmask][mask])
        zn = np.append(zn, znf[flagmask][mask])
        depth = np.append(depth, depthf[flagmask][mask])
        lande = np.append(lande, landef[flagmask][mask])
        excpot = np.append(excpot, excpotf[flagmask][mask])
        flag = np.append(flag, flagf[flagmask][mask])

    # PS. Below it applies a line depth mask, however the cut in line depth
    # should be done according to the SNR. This will be studied and implemented
    # later. E. Martioli, Aug 10 2018.

    # create mask to cutoff lines with lande g-factor without sensible values
    gmask = np.where(np.logical_and(lande > p['IC_POLAR_LSD_MIN_LANDE'], lande < p['IC_POLAR_LSD_MAX_LANDE']))
    # apply mask to the data
    wlc, zn, depth, lande = wlc[gmask], zn[gmask], depth[gmask], lande[gmask]
    excpot, flag = excpot[gmask], flag[gmask]

    if p['IC_POLAR_LSD_CCFLINES_AIR_WAVE'] :
        wlc = convert_air_to_vacuum_wl(wlc)
    
    # create mask to cutoff lines with depth lower than IC_POLAR_LSD_MIN_LINEDEPTH
    dmask = np.where(depth > p['IC_POLAR_LSD_MIN_LINEDEPTH'])
    # apply mask to the data
    wlc, zn, depth, lande = wlc[dmask], zn[dmask], depth[dmask], lande[dmask]
    excpot, flag = excpot[dmask], flag[dmask]

    loc["NUMBER_OF_LINES_USED"] = len(wlc)
    if verbose :
        print("Number of lines after filtering = ", len(wlc))

    loc["MEAN_WAVE_OF_LINES"] = np.nanmean(wlc)
    loc["MEAN_LANDE_OF_LINES"] = np.nanmean(lande)

    # calculate weights for calculation of polarimetric Z-profile
    weight = wlc * depth * lande

    # Uncomment below to print the information for the line with maximum weight
    #idxmax = np.argmax(weight)
    #print(wlc[idxmax], depth[idxmax], lande[idxmax], weight[idxmax])

    weight = weight / np.nanmax(weight)

    # store data into loc dict
    loc['LSD_LINES_WLC'] = wlc
    
    loc['LSD_LINES_ZNUMBER'] = zn
    loc['LSD_LINES_DEPTH'] = depth
    loc['LSD_LINES_POL_WEIGHT'] = weight

    loc['LSD_LINES_LANDE'] = lande
    loc['LSD_LINES_POL_EXC_POTENTIAL'] = excpot
    loc['LSD_LINES_POL_FLAG'] = flag

    # uncomment below to print out the clean line mask
    """
    print(len(wlc))
    for i in range(len(wlc)) :
        print("{0:.4f}  {1:.2f}  {2:.3f}  {3:.3f}  {4:.3f}  {5:.0f}".format(wlc[i],zn[i],depth[i],excpot[i],lande[i],flag[i]))
    """
    return loc


def get_wl_ranges(p, loc):
    """
    Function to generate a list of spectral ranges covering all spectral
    lines in the CCF mask, where the width of each individual range is
    defined by the LSD velocity vector
    
    :param p: parameter dictionary, ParamDict containing constants
        Must contain at least:
            LOG_OPT: string, option for logging
            IC_POLAR_LSD_V0: initial velocity for LSD profile
            IC_POLAR_LSD_VF: final velocity for LSD profile

    :param loc: parameter dictionary, ParamDict to store data
        loc['LSD_LINES_WLC']: numpy array (1D), central wavelengths
        
    :return loc: parameter dictionaries,
        The updated parameter dictionary adds/updates the following:
        loc['LSD_LINES_WLRANGES']: array of float pairs for wavelength ranges
       
    """
    # func_name = __NAME__ + '.get_wl_ranges()'

    # speed of light in km/s
    c = constants.c / 1000.
    # set initial and final velocity
    v0, vf = p['IC_POLAR_LSD_V0'], p['IC_POLAR_LSD_VF']
    # define vector of spectral ranges covering only regions around lines
    wlranges_tmp = []
    for w in loc['LSD_LINES_WLC']:
        dwl = w * (vf - v0) / (2. * c)
        wl0 = w - dwl
        wlf = w + dwl
        wlranges_tmp.append([wl0, wlf])
    # initialize final vector of spectral ranges
    loc['LSD_LINES_WLRANGES'] = []
    # initialize current wl0 and wlf
    current_wl0, current_wlf = wlranges_tmp[0][0], wlranges_tmp[0][1]
    # merge overlapping ranges
    for r in wlranges_tmp:
        if r[0] <= current_wlf:
            current_wlf = r[1]
        else:
            loc['LSD_LINES_WLRANGES'].append([current_wl0, current_wlf])
            current_wl0 = r[0]
            current_wlf = r[1]
    # append last range
    loc['LSD_LINES_WLRANGES'].append([current_wl0, current_wlf])

    return loc


def prepare_polarimetry_data(p, loc):
    """
    Function to prepare polarimetry data for LSD analysis.
    
    :param p: parameter dictionary, ParamDict containing constants
        Must contain at least:
            LOG_OPT: string, option for logging
            IC_POLAR_LSD_NORMALIZE: bool, normalize Stokes I data
            
    :param loc: parameter dictionary, ParamDict to store data
        Must contain at least:
            loc['WAVE']: numpy array (2D), wavelength data
            loc['STOKESI']: numpy array (2D), Stokes I data
            loc['STOKESIERR']: numpy array (2D), errors of Stokes I
            loc['POL']: numpy array (2D), degree of polarization data
            loc['POLERR']: numpy array (2D), errors of degree of polarization
            loc['NULL2']: numpy array (2D), 2nd null polarization

    :return loc: parameter dictionaries,
        The updated parameter dictionary adds/updates the following:
            loc['LSD_WAVE']: numpy array (1D), wavelength data
            loc['LSD_FLUX']: numpy array (1D), Stokes I data
            loc['LSD_FLUXERR']: numpy array (1D), errors of Stokes I
            loc['LSD_POL']: numpy array (1D), degree of polarization data
            loc['LSD_POLERR']: numpy array (1D), errors of polarization
            loc['LSD_NULL']: numpy array (1D), 2nd null polarization
        
    """

    # func_name = __NAME__ + '.prepare_polarimetry_data()'

    # get the shape of pol
    ydim, xdim = loc['POL'].shape
    
    # get wavelength ranges to be considered in each spectral order
    ordermask = get_order_ranges()
    # initialize output data vectors
    loc['LSD_WAVE'], loc['LSD_FLUX'], loc['LSD_FLUXERR'] = [], [], []
    loc['LSD_POL'], loc['LSD_POLERR'], loc['LSD_NULL'] = [], [], []
    
    # loop over each order
    for order_num in range(ydim):
        # mask NaN values
        nanmask = ~np.isnan(loc['STOKESI'][order_num])
        nanmask &= ~np.isnan(loc['POL'][order_num])
        nanmask &= ~np.isnan(loc['POLERR'][order_num])
        nanmask &= ~np.isnan(loc['STOKESIERR'][order_num])
        nanmask &= loc['STOKESI'][order_num] > 0
        nanmask &= ~np.isinf(loc['STOKESI'][order_num])
        nanmask &= ~np.isinf(loc['POL'][order_num])
        nanmask &= ~np.isinf(loc['POLERR'][order_num])
        nanmask &= ~np.isinf(loc['STOKESIERR'][order_num])
        
        wl_tmp = loc['WAVE'][order_num][nanmask]
        pol_tmp = loc['POL'][order_num][nanmask]
        polerr_tmp = loc['POLERR'][order_num][nanmask]
        flux_tmp = loc['STOKESI'][order_num][nanmask]
        fluxerr_tmp = loc['STOKESIERR'][order_num][nanmask]
        null_tmp = loc['NULL2'][order_num][nanmask]

        # set order wavelength limits
        wl0, wlf = ordermask[order_num][0], ordermask[order_num][1]
        # create wavelength mask
        mask = np.where(np.logical_and(wl_tmp > wl0, wl_tmp < wlf))

        # test if order is not empty
        if len(wl_tmp[mask]):
            # get masked data
            wl, flux, fluxerr = wl_tmp[mask], flux_tmp[mask], fluxerr_tmp[mask]
            pol, polerr, null = pol_tmp[mask], polerr_tmp[mask], null_tmp[mask]

            if p['IC_POLAR_LSD_NORMALIZE']:
                # measure continuum
                # TODO: Should be in constant file
                kwargs = dict(binsize=80, overlap=15, window=3,
                              mode='max', use_linear_fit=True)
                cont, xbin, ybin = continuum(wl, flux, **kwargs)
                # normalize flux
                flux = flux / cont
                fluxerr = fluxerr / cont

            # append data to output vector
            loc['LSD_WAVE'] = np.append(loc['LSD_WAVE'], wl)
            loc['LSD_FLUX'] = np.append(loc['LSD_FLUX'], flux)
            loc['LSD_FLUXERR'] = np.append(loc['LSD_FLUXERR'], fluxerr)
            loc['LSD_POL'] = np.append(loc['LSD_POL'], pol)
            loc['LSD_POLERR'] = np.append(loc['LSD_POLERR'], polerr)
            loc['LSD_NULL'] = np.append(loc['LSD_NULL'], null)

    return loc


def lsd_analysis(p, loc):
    """
    Function to perform Least Squares Deconvolution (LSD) analysis.
    
    :param p: parameter dictionary, ParamDict containing constants
        Must contain at least:
            LOG_OPT: string, option for logging
            
    :param loc: parameter dictionary, ParamDict to store data
        Must contain at least:
            loc['IC_POLAR_LSD_V0']: initial velocity for LSD profile
            loc['IC_POLAR_LSD_VF']: final velocity for LSD profile
            loc['IC_POLAR_LSD_NP']: number of points in the LSD profile
            loc['LSD_WAVE']: numpy array (1D), wavelength data
            loc['LSD_STOKESI']: numpy array (1D), Stokes I data
            loc['LSD_STOKESIERR']: numpy array (1D), errors of Stokes I
            loc['LSD_POL']: numpy array (1D), degree of polarization data
            loc['LSD_POLERR']: numpy array (1D), errors of polarization
            loc['LSD_NULL']: numpy array (1D), 2nd null polarization
            loc['LSD_LINES_WLC']: numpy array (1D), central wavelengths
            loc['LSD_LINES_DEPTH']: numpy array (1D), line depths
            loc['LSD_LINES_POL_WEIGHT']: numpy array (1D), line weights =
                                         depth * lande * wlc
        
    :return loc: parameter dictionaries,
        The updated parameter dictionary adds/updates the following:
            loc['LSD_VELOCITIES']: numpy array (1D), LSD profile velocities
            loc['LSD_STOKESI']: numpy array (1D), LSD profile for Stokes I
            loc['LSD_STOKESI_MODEL']: numpy array (1D), LSD gaussian model
                                      profile for Stokes I
            loc['LSD_STOKESVQU']: numpy array (1D), LSD profile for Stokes
                                  Q,U,V polarimetry spectrum
            loc['LSD_NULL']: numpy array (1D), LSD profile for null
                                  polarization spectrum
        
    """

    # func_name = __NAME__ + '.lsd_analysis()'

    # initialize variables to define velocity vector of output LSD profile
    v0, vf, m = p['IC_POLAR_LSD_V0'], p['IC_POLAR_LSD_VF'], p['IC_POLAR_LSD_NP']

    # create velocity vector for output LSD profile
    loc['LSD_VELOCITIES'] = np.linspace(v0, vf, m)

    # create line pattern matrix for flux LSD
    mm, mmp = line_pattern_matrix(loc['LSD_WAVE'], loc['LSD_LINES_WLC'],
                                  loc['LSD_LINES_DEPTH'],
                                  loc['LSD_LINES_POL_WEIGHT'],
                                  loc['LSD_VELOCITIES'])

    # calculate flux LSD profile
    loc['LSD_STOKESI'], loc['LSD_STOKESI_ERR'] = calculate_lsd_profile(loc['LSD_WAVE'],
                                               1.0 - loc['LSD_FLUX'],
                                               loc['LSD_FLUXERR'],
                                               loc['LSD_VELOCITIES'], mm,
                                               normalize=False)

    # calculate model Stokes I spectrum
    loc['LSD_STOKESI_MODEL'] = 1.0 - mm.dot(loc['LSD_STOKESI'])
    
    # uncomment below to check model Stokes I spectrum
    #plt.plot(loc['LSD_WAVE'],loc['LSD_FLUX'])
    #plt.plot(loc['LSD_WAVE'],loc['LSD_STOKESI_MODEL'])
    #plt.show()
    
    # return profile to standard natural shape (as absorption)
    loc['LSD_STOKESI'] = 1.0 - loc['LSD_STOKESI']
    
    # fit gaussian to the measured flux LSD profile
    loc['LSD_STOKESI_MODEL'], loc['LSD_FIT_RV'], loc[
        'LSD_FIT_RESOL'] = fit_gaussian_to_lsd_profile(loc['LSD_VELOCITIES'],
                                                       loc['LSD_STOKESI'])

    # calculate polarimetry LSD profile
    loc['LSD_STOKESVQU'], loc['LSD_STOKESVQU_ERR'] = calculate_lsd_profile(loc['LSD_WAVE'],
                                                 loc['LSD_POL'],
                                                 loc['LSD_POLERR'],
                                                 loc['LSD_VELOCITIES'], mmp)

    # calculate model Stokes VQU spectrum
    loc['LSD_POL_MODEL'] = mmp.dot(loc['LSD_STOKESVQU'])

    # uncomment below to check model Stokes VQU spectrum
    #plt.errorbar(loc['LSD_WAVE'], loc['LSD_POL'], yerr=loc['LSD_POLERR'], fmt='.')
    #plt.plot(loc['LSD_WAVE'], loc['LSD_POL_MODEL'], '-')
    #plt.show()
    
    # calculate null polarimetry LSD profile
    loc['LSD_NULL'], loc['LSD_NULL_ERR'] = calculate_lsd_profile(loc['LSD_WAVE'], loc['LSD_NULL'],
                                            loc['LSD_POLERR'],
                                            loc['LSD_VELOCITIES'], mmp)

                                            
    # make sure output arrays are numpy arrays
    if p['IC_POLAR_LSD_REMOVE_EDGES'] :
        loc['LSD_VELOCITIES'] = loc['LSD_VELOCITIES'][1:-2]
        loc['LSD_STOKESVQU'] = np.array(loc['LSD_STOKESVQU'][1:-2])
        loc['LSD_STOKESVQU_ERR'] = np.array(loc['LSD_STOKESVQU_ERR'][1:-2])
        loc['LSD_STOKESI'] = np.array(loc['LSD_STOKESI'][1:-2])
        loc['LSD_STOKESI_ERR'] = np.array(loc['LSD_STOKESI_ERR'][1:-2])
        loc['LSD_NULL'] = np.array(loc['LSD_NULL'][1:-2])
        loc['LSD_NULL_ERR'] = np.array(loc['LSD_NULL_ERR'][1:-2])
        loc['LSD_STOKESI_MODEL'] = np.array(loc['LSD_STOKESI_MODEL'][1:-2])
    else :
        loc['LSD_STOKESVQU'] = np.array(loc['LSD_STOKESVQU'])
        loc['LSD_STOKESVQU_ERR'] = np.array(loc['LSD_STOKESVQU_ERR'])
        loc['LSD_STOKESI'] = np.array(loc['LSD_STOKESI'])
        loc['LSD_STOKESI_ERR'] = np.array(loc['LSD_STOKESI_ERR'])
        loc['LSD_NULL'] = np.array(loc['LSD_NULL'])
        loc['LSD_NULL_ERR'] = np.array(loc['LSD_NULL_ERR'])
        loc['LSD_STOKESI_MODEL'] = np.array(loc['LSD_STOKESI_MODEL'])

    # calculate statistical quantities
    loc['LSD_POL_MEAN'] = np.nanmean(loc['LSD_POL'])
    loc['LSD_POL_STDDEV'] = np.nanstd(loc['LSD_POL'])
    loc['LSD_POL_MEDIAN'] = np.nanmedian(loc['LSD_POL'])
    loc['LSD_POL_MEDABSDEV'] = np.nanmedian(np.abs(loc['LSD_POL'] -
                                                loc['LSD_POL_MEDIAN']))
    loc['LSD_STOKESVQU_MEAN'] = np.nanmean(loc['LSD_STOKESVQU'])
    loc['LSD_STOKESVQU_STDDEV'] = np.nanstd(loc['LSD_STOKESVQU'])
    loc['LSD_NULL_MEAN'] = np.nanmean(loc['LSD_NULL'])
    loc['LSD_NULL_STDDEV'] = np.nanstd(loc['LSD_NULL'])

    return loc


def line_pattern_matrix(wl, wlc, depth, weight, vels):
    """
    Function to calculate the line pattern matrix M given in Eq (4) of paper
    Donati et al. (1997), MNRAS 291, 658-682
    
    :param wl: numpy array (1D), input wavelength data (size n = spectrum size)
    :param wlc: numpy array (1D), central wavelengths (size = number of lines)
    :param depth: numpy array (1D), line depths (size = number of lines)
    :param weight: numpy array (1D), line polar weights (size = number of lines)
    :param vels: numpy array (1D), , LSD profile velocity vector (size = m)
    
    :return mm, mmp
        mm: numpy array (2D) of size n x m, line pattern matrix for flux LSD.
        mmp: numpy array (2D) of size n x m, line pattern matrix for polar LSD.
    """

    # set number of points and velocity (km/s) limits in LSD profile
    m, v0, vf = len(vels), vels[0], vels[-1]

    # speed of light in km/s
    c = constants.c / 1000.

    # set number of spectral points
    n = len(wl)

    # initialize line pattern matrix for flux LSD
    mm = np.zeros((n, m))

    # initialize line pattern matrix for polar LSD
    mmp = np.zeros((n, m))

    # set values of line pattern matrix M
    for l in range(len(wlc)):
    
        wl_prof = wlc[l] * (1. + vels / c)
        
        line, = np.where(np.logical_and(wl >= wl_prof[0], wl <= wl_prof[-1]))

        # Calculate line velocities for the observed wavelength sampling: v = c Δλ / λ
        vl = c * (wl[line] - wlc[l]) / wlc[l]
        
        for i in range(len(line)) :
            '''
                #Use the nearest neighbor point
                j_prof = np.argmin(np.abs(vels - vl[i]))
                mmp[i][j_prof] += weight[l]
                mm[i][j_prof] += depth[l]
              '''
            #Linear interpolation:
            j_prof = np.searchsorted(vels, vl[i], side='right')
            vel_weight = (vl[i] - vels[j_prof-1]) / (vels[j_prof] - vels[j_prof-1])
            mmp[line[i]][j_prof-1] += weight[l] * (1. - vel_weight)
            mmp[line[i]][j_prof] += weight[l] * vel_weight
            mm[line[i]][j_prof-1] += depth[l] * (1. - vel_weight)
            mm[line[i]][j_prof] += depth[l] * vel_weight

    return mm, mmp


def calculate_lsd_profile(wl, flux, fluxerr, vels, mm, normalize=False):
    """
    Function to calculate the LSD profile Z given in Eq (4) of paper
    Donati et al. (1997), MNRAS 291, 658-682
    
    :param wl: numpy array (1D), input wavelength data (size = n)
    :param flux: numpy array (1D), input flux or polarimetry data (size = n)
    :param fluxerr: numpy array (1D), input flux or polarimetry error data
                    (size = n)
    :param vels: numpy array (1D), , LSD profile velocity vector (size = m)
    :param mm: numpy array (2D) of size n x m, line pattern matrix for LSD.
    :param normalize: bool, to calculate a continuum and normalize profile
    
    :return Z: numpy array (1D) of size m, LSD profile.
    """

    # set number of spectral points
    # noinspection PyUnusedLocal
    n = len(wl)

    # First calculate transpose of M
    mmt = np.matrix.transpose(mm)

    # Initialize matrix for dot product between MT . S^2
    mmt_x_s2 = np.zeros_like(mmt)

    # Then calculate dot product between MT . S^2, where S^2=covariance matrix
    for j in range(np.shape(mmt)[0]):
        mmt_x_s2[j] = mmt[j] / (fluxerr * fluxerr)

    # calculate autocorrelation, i.e., MT . S^2 . M
    mmt_x_s2_x_mm = mmt_x_s2.dot(mm)

    # calculate the inverse of autocorrelation using numpy pinv method
    mmt_x_s2_x_mm_inv = np.linalg.pinv(mmt_x_s2_x_mm)

    # calculate cross correlation term, i.e. MT . S^2 . Y
    x_corr_term = mmt_x_s2.dot(flux)
    
    # calculate velocity profile
    zz = mmt_x_s2_x_mm_inv.dot(x_corr_term)

    # calculate error of velocity profile
    zz_err = np.sqrt(np.diag(mmt_x_s2_x_mm_inv))

    if normalize:
        # calculate continuum of LSD profile to remove trend
        cont_z, xbin, ybin = continuum(vels, zz, binsize=20,
                                                  overlap=5,
                                                  sigmaclip=3.0, window=2,
                                                  mode="median",
                                                  use_linear_fit=False)
                                                  
                                                  
                                                  
        # calculate normalized and detrended LSD profile
        zz /= cont_z
        zz_err /= cont_z

    return zz, zz_err


def gauss_function(x, a, x0, sigma, dc):
    return a * np.exp(-0.5 * ((x - x0) / sigma) ** 2) + dc

def fit_gaussian_to_lsd_profile(vels, zz, verbose=False):
    """
        Function to fit gaussian to LSD Stokes I profile.
        
        :param vels: numpy array (1D), input velocity data
        :param zz: numpy array (1D), input LSD profile data
        
        :return z_gauss, RV, resolving_power:
            z_gauss: numpy array (1D), gaussian fit to LSD profile (same size
                    as input vels and Z)
            RV: float, velocity of minimum obtained from gaussian fit
            resolving_power: float, spectral resolving power calculated from
                            sigma of gaussian fit
        """

    # set speed of light in km/s
    c = constants.c / 1000.

    # obtain velocity at minimum, amplitude, and sigma for initial guess
    rvel = vels[np.argmin(zz)]
    y0 = np.median(zz)
    amplitude = np.abs(y0 - np.min(zz))
    resolving_power = 50000.
    sig = c / (resolving_power * 2.3548)
    
    # get inverted profile
    z_inv = 1.0 - zz

    # fit gaussian profile
    guess = [amplitude, rvel, sig, y0]
    #guess = [0.1, 0.0, 1.0]
    
    # noinspection PyTypeChecker
    try :
        popt, pcov = curve_fit(gauss_function, vels, z_inv, p0=guess)
    except :
        if verbose :
            # log start of LSD analysis calculations
            wmsg = ': failed to fit gaussian to LSD profile'
            print('WARNING', wmsg)
        popt = guess
    # initialize output profile vector
    z_gauss = np.zeros_like(vels)

    for i in range(len(z_gauss)):
        # calculate gaussian model profile
        z_gauss[i] = gauss_function(vels[i], *popt)

    # invert fit profile
    z_gauss = 1.0 - z_gauss

    # calculate full width at half maximum (fwhm)
    fwhm = 2.35482 * popt[2]
    # calculate resolving power from mesasured fwhm
    resolving_power = c / fwhm

    # set radial velocity directly from fitted v_0
    rv = popt[1]

    return z_gauss, rv, resolving_power


def get_order_ranges():
    """
    Function to provide the valid wavelength ranges for each order in SPIrou.
    
    :param: None
    
    :return orders: array of float pairs for wavelength ranges
    """
    # TODO: Should be moved to file in .../INTROOT/SpirouDRS/data/
    orders = [[963.6, 986.0], [972.0, 998.4], [986.3, 1011], [1000.1, 1020],
              [1015, 1035], [1027.2, 1050], [1042, 1065], [1055, 1078],
              [1070, 1096],
              [1084, 1112.8], [1098, 1128], [1113, 1146], [1131, 1162],
              [1148, 1180],
              [1166, 1198], [1184, 1216], [1202, 1235], [1222, 1255],
              [1243, 1275],
              [1263, 1297], [1284, 1320], [1306, 1342], [1328, 1365],
              [1352, 1390],
              [1377, 1415], [1405, 1440], [1429, 1470], [1456, 1497],
              [1485, 1526],
              [1515, 1557], [1545, 1590], [1578, 1623], [1609, 1657],
              [1645, 1692],
              [1681, 1731], [1722, 1770], [1760, 1810], [1800, 1855],
              [1848, 1900],
              [1890, 1949], [1939, 1999], [1991, 2050], [2044.5, 2105],
              [2104, 2162],
              [2161, 2226], [2225, 2293], [2291, 2362], [2362, 2430],
              [2440, 2510]]
    return orders


def polar_lsd_plot(p, loc):
    plot_name = 'polar_lsd_plot'
    # get data from loc
    vels = loc['LSD_VELOCITIES']
    zz = loc['LSD_STOKESI']
    zz_err = loc['LSD_STOKESI_ERR']
    zgauss = loc['LSD_STOKESI_MODEL']
    z_p = loc['LSD_STOKESVQU']
    z_p_err = loc['LSD_STOKESVQU_ERR']
    z_np = loc['LSD_NULL']
    z_np_err = loc['LSD_NULL_ERR']
    stokes = loc['STOKES']

    # ---------------------------------------------------------------------
    # set up fig
    fig, frames = setup_figure(p, ncols=1, nrows=3)
    # clear the current figure
    #plt.clf()

    # ---------------------------------------------------------------------
    frame = frames[0]
    frame.errorbar(vels, zz, yerr=zz_err, fmt='.', color='red')
    frame.plot(vels, zz, '-', linewidth=0.3, color='red')
    frame.plot(vels, zgauss, '-', color='green')
    title = 'LSD Analysis'
    ylabel = 'Stokes I profile'
    xlabel = ''
    # set title and labels
    frame.set(title=title, xlabel=xlabel, ylabel=ylabel)
    # ---------------------------------------------------------------------

    # ---------------------------------------------------------------------
    frame = frames[1]
    title = ''
    frame.errorbar(vels, z_p, yerr=z_p_err, fmt='.', color='blue')
    frame.plot(vels, z_p, '-', linewidth=0.5, color='blue')
    ylabel = 'Stokes {0} profile'.format(stokes)
    xlabel = ''
    # set title and labels
    frame.set(title=title, xlabel=xlabel, ylabel=ylabel)
    plot_y_lims = frame.get_ylim()
    y_range = np.abs(plot_y_lims[1] - plot_y_lims[0])
    # ---------------------------------------------------------------------

    # ---------------------------------------------------------------------
    frame = frames[2]
    bottom, top = loc['LSD_NULL_MEAN'] - y_range/2.0, loc['LSD_NULL_MEAN'] + y_range/2.0
    frame.set_ylim(bottom, top)
    frame.errorbar(vels, z_np, yerr=z_np_err, fmt='.', color='orange')
    frame.plot(vels, z_np, '-', linewidth=0.5, color='orange')
    xlabel = 'velocity (km/s)'
    ylabel = 'Null profile'
    # set title and labels
    frame.set(title=title, xlabel=xlabel, ylabel=ylabel)
    # ---------------------------------------------------------------------
    plt.show()


def save_lsd_fits(filename, loc, p) :
    """
    Function to save output FITS image to store LSD analyis.
    
    :param filename: string, Output FITS filename
    :param loc: parameter dictionary, ParamDict to store data
        Must contain at least:
        loc['LSDFITRV']: float, RV from LSD gaussian fit (km/s)
        loc['LSD_VELOCITIES']: numpy array (2D), LSD analysis data
        loc['LSD_STOKESVQU']: numpy array (2D), LSD analysis data
        loc['LSD_STOKESVQU_ERR']: numpy array (2D), LSD analysis data
        loc['LSD_STOKESI']: numpy array (2D), LSD analysis data
        loc['LSD_STOKESI_ERR']: numpy array (2D), LSD analysis data
        loc['LSD_STOKESI_MODEL']: numpy array (2D), LSD analysis data
        loc['LSD_NULL']: numpy array (2D), LSD analysis data
    """

    header = loc['HEADER0']
    header1 = loc['HEADER1']
    
    header.set('ORIGIN', "spirou_lsd")

    header.set('LSDFITRV', loc['LSD_FIT_RV'], 'RV from LSD gaussian fit (km/s)')
    
    header.set('POLAVG', loc['LSD_POL_MEAN'], 'Mean degree of polarization')
    header.set('POLSTD', loc['LSD_POL_STDDEV'], 'Std deviation of degree of polarization')
    header.set('POLMED', loc['LSD_POL_MEDIAN'], 'Median degree of polarization')
    header.set('POLMEDEV', loc['LSD_POL_MEDABSDEV'], 'Median deviations of degree of polarization')
    header.set('LSDPAVG', loc['LSD_STOKESVQU_MEAN'], 'Mean of Stokes VQU LSD profile')
    header.set('LSDPSTD', loc['LSD_STOKESVQU_STDDEV'], 'Std deviation of Stokes VQU LSD profile')
    header.set('LSDNAVG', loc['LSD_NULL_MEAN'], 'Mean of Stokes VQU LSD null profile')
    header.set('LSDNSTD', loc['LSD_NULL_STDDEV'], 'Std deviation of Stokes VQU LSD null profile')
    
    header.set('MASKFILE', os.path.basename(loc['LSD_MASK_FILE']), 'Mask file used in LSD analysis')
    header.set('NLINMASK', loc["NUMBER_OF_LINES_IN_MASK"], 'Number of lines in the original mask')
    header.set('NLINUSED', loc["NUMBER_OF_LINES_USED"], 'Number of lines used in LSD analysis')
    header.set('WAVEAVG', loc["MEAN_WAVE_OF_LINES"], 'Mean wavelength of lines used in LSD analysis')
    header.set('LANDEAVG', loc["MEAN_LANDE_OF_LINES"], 'Mean lande of lines used in LSD analysis')

    """
    idx = 0
    for key in p.keys() :
        param_key = "PARAM{:03d}".format(idx)
        header.set(param_key, key, str(p[key]))
        #print(param_key, p[key], key)
        idx += 1
    """
    primary_hdu = fits.PrimaryHDU(header=header)

    hdu_vels = fits.ImageHDU(data=loc['LSD_VELOCITIES'], name="Velocity", header=header1)
    hdu_pol = fits.ImageHDU(data=loc['LSD_STOKESVQU'], name="StokesVQU")
    hdu_pol_err = fits.ImageHDU(data=loc['LSD_STOKESVQU_ERR'], name="StokesVQU_Err")
    hdu_flux = fits.ImageHDU(data=loc['LSD_STOKESI'], name="StokesI")
    hdu_flux_err = fits.ImageHDU(data=loc['LSD_STOKESI_ERR'], name="StokesI_Err")
    hdu_fluxmodel = fits.ImageHDU(data=loc['LSD_STOKESI_MODEL'], name="StokesIModel")
    hdu_null = fits.ImageHDU(data=loc['LSD_NULL'], name="Null")
    hdu_null_err = fits.ImageHDU(data=loc['LSD_NULL_ERR'], name="Null_Err")

    mef_hdu = fits.HDUList([primary_hdu, hdu_vels, hdu_pol, hdu_pol_err, hdu_flux, hdu_flux_err, hdu_fluxmodel, hdu_null, hdu_null_err])

    mef_hdu.writeto(filename, overwrite=True)


def select_lsd_mask(p, loc) :
    """
        Function to select a LSD mask file from a given list of
        mask repositories that best match the object temperature.
        
        :param p: parameter dictionary,
        Must contain at least:
        p['OBJTEMP']: float, object effective temperature
        p['IC_POLAR_LSD_MASKS_REPOSITORIES']: list, array of strings
        containing paths to LSD mask repositories to search for masks
        
        return: string, file path for the LSD mask that best match Teff
        """
    
    file_list = []
    
    for repo in p['IC_POLAR_LSD_MASKS_REPOSITORIES'] :
        repo_full_path = os.path.dirname(__file__) + "/" + repo
        file_list += glob.glob(repo_full_path)
    
    obj_Temperature = loc['OBJTEMP']

    #print("Selecting mask for Teff=",obj_Temperature)

    mindiff = 1e20
    selected_mask = os.path.dirname(__file__) + '/lsd_masks/marcs_t3000g50_all'

    for filepath in file_list :
        basename = os.path.basename(filepath)
    
        if basename[:5] == "marcs" :
            #assuming the following format: marcs_t3000g50_all
            mask_temp = float(basename[7:11])
        elif basename[0] == 't':
            #assuming the following format: t4000_g4.0_m0.00
            mask_temp = float(basename[1:5])
        else :
            #print("WARNING: mask file {0} not recognized, skipping ...".format(basename))
            continue

        tdiff = np.abs(mask_temp - obj_Temperature)

        #print("Mask file: {0} T={1}".format(basename,mask_temp))

        if tdiff < mindiff :
            selected_mask = filepath
            mindiff = tdiff

    #print("Selectect mask:", selected_mask)
    
    return selected_mask

#### Function to detect continuum #########
def continuum(x, y, binsize=200, overlap=100, sigmaclip=3.0, window=3,
              mode="median", use_linear_fit=False, telluric_bands=[], outx=None):
    """
    Function to calculate continuum
    :param x,y: numpy array (1D), input data (x and y must be of the same size)
    :param binsize: int, number of points in each bin
    :param overlap: int, number of points to overlap with adjacent bins
    :param sigmaclip: int, number of times sigma to cut-off points
    :param window: int, number of bins to use in local fit
    :param mode: string, set combine mode, where mode accepts "median", "mean",
                 "max"
    :param use_linear_fit: bool, whether to use the linar fit
    :param telluric_bands: list of float pairs, list of IR telluric bands, i.e,
                           a list of wavelength ranges ([wl0,wlf]) for telluric
                           absorption
    
    :return continuum, xbin, ybin
        continuum: numpy array (1D) of the same size as input arrays containing
                   the continuum data already interpolated to the same points
                   as input data.
        xbin,ybin: numpy arrays (1D) containing the bins used to interpolate
                   data for obtaining the continuum
    """

    if outx is None :
        outx = x
    
    # set number of bins given the input array length and the bin size
    nbins = int(np.floor(len(x) / binsize)) + 1

    # initialize arrays to store binned data
    xbin, ybin = [], []
    
    for i in range(nbins):
        # get first and last index within the bin
        idx0 = i * binsize - overlap
        idxf = (i + 1) * binsize + overlap
        # if it reaches the edges then reset indexes
        if idx0 < 0:
            idx0 = 0
        if idxf >= len(x):
            idxf = len(x) - 1
        # get data within the bin
        xbin_tmp = np.array(x[idx0:idxf])
        ybin_tmp = np.array(y[idx0:idxf])

        # create mask of telluric bands
        telluric_mask = np.full(np.shape(xbin_tmp), False, dtype=bool)
        for band in telluric_bands :
            telluric_mask += (xbin_tmp > band[0]) & (xbin_tmp < band[1])

        # mask data within telluric bands
        xtmp = xbin_tmp[~telluric_mask]
        ytmp = ybin_tmp[~telluric_mask]
        
        # create mask to get rid of NaNs
        nanmask = np.logical_not(np.isnan(ytmp))
        
        if i == 0 and not use_linear_fit:
            xbin.append(x[0] - np.abs(x[1] - x[0]))
            # create mask to get rid of NaNs
            localnanmask = np.logical_not(np.isnan(y))
            ybin.append(np.median(y[localnanmask][:binsize]))
        
        if len(xtmp[nanmask]) > 2 :
            # calculate mean x within the bin
            xmean = np.mean(xtmp[nanmask])
            # calculate median y within the bin
            medy = np.median(ytmp[nanmask])

            # calculate median deviation
            medydev = np.median(np.absolute(ytmp[nanmask] - medy))
            # create mask to filter data outside n*sigma range
            filtermask = (ytmp[nanmask] > medy) & (ytmp[nanmask] < medy +
                                                   sigmaclip * medydev)
            if len(ytmp[nanmask][filtermask]) > 2:
                # save mean x wihthin bin
                xbin.append(xmean)
                if mode == 'max':
                    # save maximum y of filtered data
                    ybin.append(np.max(ytmp[nanmask][filtermask]))
                elif mode == 'median':
                    # save median y of filtered data
                    ybin.append(np.median(ytmp[nanmask][filtermask]))
                elif mode == 'mean':
                    # save mean y of filtered data
                    ybin.append(np.mean(ytmp[nanmask][filtermask]))
                else:
                    emsg = 'Can not recognize selected mode="{0}"...exiting'
                    print('error', emsg.format(mode))

        if i == nbins - 1 and not use_linear_fit:
            xbin.append(x[-1] + np.abs(x[-1] - x[-2]))
            # create mask to get rid of NaNs
            localnanmask = np.logical_not(np.isnan(y[-binsize:]))
            ybin.append(np.median(y[-binsize:][localnanmask]))

    # Option to use a linearfit within a given window
    if use_linear_fit:
        # initialize arrays to store new bin data
        newxbin, newybin = [], []

        # loop around bins to obtain a linear fit within a given window size
        for i in range(len(xbin)):
            # set first and last index to select bins within window
            idx0 = i - window
            idxf = i + 1 + window
            # make sure it doesnt go over the edges
            if idx0 < 0: idx0 = 0
            if idxf > nbins: idxf = nbins - 1

            # perform linear fit to these data
            slope, intercept, r_value, p_value, std_err = stats.linregress(xbin[idx0:idxf], ybin[idx0:idxf])

            if i == 0 :
                # append first point to avoid crazy behaviours in the edge
                newxbin.append(x[0] - np.abs(x[1] - x[0]))
                newybin.append(intercept + slope * newxbin[0])
            
            # save data obtained from the fit
            newxbin.append(xbin[i])
            newybin.append(intercept + slope * xbin[i])

            if i == len(xbin) - 1 :
                # save data obtained from the fit
                newxbin.append(x[-1] + np.abs(x[-1] - x[-2]))
                newybin.append(intercept + slope * newxbin[-1])

        xbin, ybin = newxbin, newybin

    # interpolate points applying an Spline to the bin data
    sfit = UnivariateSpline(xbin, ybin, s=0)
    #sfit.set_smoothing_factor(0.5)
    
    # Resample interpolation to the original grid
    cont = sfit(outx)

    # return continuum and x and y bins
    return cont, xbin, ybin
    ##-- end of continuum function


def nrefrac(wavelength, density=1.0):
   """Calculate refractive index of air from Cauchy formula.

   Input: wavelength in nm, density of air in amagat (relative to STP,
   e.g. ~10% decrease per 1000m above sea level).
   Returns N = (n-1) * 1.e6.
   """

   # The IAU standard for conversion from air to vacuum wavelengths is given
   # in Morton (1991, ApJS, 77, 119). For vacuum wavelengths (VAC) in
   # Angstroms, convert to air wavelength (AIR) via:

   #  AIR = VAC / (1.0 + 2.735182E-4 + 131.4182 / VAC^2 + 2.76249E8 / VAC^4)

   wl2inv = (1.e3/wavelength)**2
   refracstp = 272.643 + 1.2288 * wl2inv  + 3.555e-2 * wl2inv**2
   return density * refracstp


def convert_vacuum_to_air_wl(vacuum_wavelength, air_density=1.0) :
    air_wavelength = vacuum_wavelength / ( 1. + 1.e-6 * nrefrac(vacuum_wavelength, density=air_density))
    return air_wavelength


def convert_air_to_vacuum_wl(air_wavelength, air_density=1.0) :
    vacuum_wavelength = air_wavelength * ( 1. + 1.e-6 * nrefrac(air_wavelength, density=air_density))
    return vacuum_wavelength

def setup_figure(p, figsize=(10, 8), ncols=1, nrows=1, attempt=0):
    """
    Extra steps to setup figure. On some OS getting error

    "TclError" when using TkAgg. A possible solution to this is to
    try switching to Agg

    :param p:
    :param figsize:
    :param ncols:
    :param nrows:
    :return:
    """
    func_name = __NAME__ + '.setup_figure()'
    fix = True
    while fix:
        if ncols == 0 and nrows == 0:
            try:
                fig = plt.figure()
                plt.clf()
                return fig
            except Exception as e:
                if fix:
                    attempt_tcl_error_fix()
                    fix = False
                else:
                    emsg1 = 'An matplotlib error occured'
                    emsg2 = '\tBackend = {0}'.format(plt.get_backend())
                    emsg3 = '\tError {0}: {1}'.format(type(e), e)
                    print(p, 'error', [emsg1, emsg2, emsg3])
        else:
            try:
                fig, frames = plt.subplots(ncols=ncols, nrows=nrows,
                                           figsize=figsize)
                return fig, frames
            except Exception as e:
                if fix:
                    attempt_tcl_error_fix()
                    fix = False
                else:
                    emsg1 = 'An matplotlib error occured'
                    emsg2 = '\tBackend = {0}'.format(plt.get_backend())
                    emsg3 = '\tError {0}: {1}'.format(type(e), e)
                    print('error', [emsg1, emsg2, emsg3])

    if attempt == 0:
        return setup_figure(p, figsize=figsize, ncols=ncols, nrows=nrows,
                            attempt=1)
    else:
        emsg1 = 'Problem with matplotlib figure/frame setup'
        emsg2 = '\tfunction = {0}'.format(func_name)
        print('error', [emsg1, emsg2])
