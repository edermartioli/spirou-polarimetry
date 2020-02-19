#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Spirou polarimetry module

Created on 2018-06-12 at 9:31

@author: E. Martioli

"""
from __future__ import division
import numpy as np
import os

import spirouLSD

import astropy.io.fits as fits
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from scipy import interpolate
from scipy.interpolate import UnivariateSpline
from scipy import stats
from scipy import constants

from copy import copy, deepcopy

# =============================================================================
# Define variables
# =============================================================================
# Name of program
__NAME__ = 'spirouPolar.py'
# -----------------------------------------------------------------------------


# =============================================================================
# Define user functions
# =============================================================================
def sort_polar_files(p):
    """
    Function to sort input data for polarimetry.
        
    :param p: parameter dictionary, ParamDict containing constants
        Must contain at least:
            LOG_OPT: string, option for logging
            REDUCED_DIR: string, directory path where reduced data are stored
            ARG_FILE_NAMES: list, list of input filenames
            KW_CMMTSEQ: string, FITS keyword where to find polarimetry
                        information
                        
    :return polardict: dictionary, ParamDict containing information on the
                       input data
                       adds an entry for each filename, each entry is a
                       dictionary containing:
                       - basename, hdr, cdr, exposure, stokes, fiber, data
                       for each file
    """

    func_name = __NAME__ + '.sort_polar_files()'

    polardict = {}
    
    # set default properties
    stokes, exposure, expstatus = 'UNDEF', 0, False

    # loop over all input files
    for filename in p['INPUT_FILES']:
        
        # initialize dictionary to store data for this file
        polardict[filename] = {}
        
        # Get t.fits and v.fits files if they exist
        tfits = filename.replace("e.fits","t.fits")
        polardict[filename]["TELLURIC_REDUC_FILENAME"] = ""
        if os.path.exists(tfits):
            polardict[filename]["TELLURIC_REDUC_FILENAME"] = tfits
            wmsg = 'Telluric file {0} loaded successfully'
            wargs = [tfits]
            print('info', wmsg.format(*wargs))
    
        vfits = filename.replace("e.fits","v.fits")
        polardict[filename]["RV_FILENAME"] = ""
        if os.path.exists(vfits) and p['IC_POLAR_SOURCERV_CORRECT'] :
            vhdr = fits.getheader(vfits)
            polardict[filename]["RV_FILENAME"] = vfits
            polardict[filename]["SOURCE_RV"] = float(vhdr['CCFRV'])
            wmsg = 'CCF RV={0:.5f} km/s from file {1} loaded successfully'
            wargs = [polardict[filename]["SOURCE_RV"], vfits]
            print('info', wmsg.format(*wargs))
        else :
            polardict[filename]["SOURCE_RV"] = 0.0
    
        # load SPIRou spectrum
        hdu = fits.open(filename)
        hdr = hdu[0].header
        hdr1 = hdu[1].header

        polardict[filename]["BERV"] = hdr1['BERV']

        # ------------------------------------------------------------------
        # add filepath
        polardict[filename]["filepath"] = os.path.abspath(filename)
        # add filename
        polardict[filename]["basename"] = os.path.basename(filename)
        
        # try to get polarisation header key
        if 'CMMTSEQ' in hdr and hdr['CMMTSEQ'] != "":
            cmmtseq = hdr['CMMTSEQ'].split(" ")
            stokes, exposure = cmmtseq[0], int(cmmtseq[2][0])
            expstatus = True
        else:
            exposure += 1
            wmsg = 'File {0} has empty key="CMMTSEQ", setting Stokes={1} Exposure={2}'
            wargs = [filename, stokes, exposure]
            
            print('warning', wmsg.format(*wargs))
            expstatus = False

        # store exposure number
        polardict[filename]["exposure"] = exposure
        # store stokes parameter
        polardict[filename]["stokes"] = stokes

        # ------------------------------------------------------------------
        # log file addition
        wmsg = 'File {0}: Stokes={1} exposure={2}'
        wargs = [filename, stokes, str(exposure)]
        print('info', wmsg.format(*wargs))

    # return polarDict
    return polardict


def load_data(p, polardict, loc):
    """
    Function to load input SPIRou data for polarimetry.
        
    :param p: parameter dictionary, ParamDict containing constants
        Must contain at least:
            LOG_OPT: string, option for logging
            IC_POLAR_STOKES_PARAMS: list, list of stokes parameters
            IC_POLAR_FIBERS: list, list of fiber types used in polarimetry
    
    :param polardict: dictionary, ParamDict containing information on the
                      input data
        
    :param loc: parameter dictionary, ParamDict to store data
        
    :return p, loc: parameter dictionaries,
        The updated parameter dictionary adds/updates the following:
            FIBER: saves reference fiber used for base file in polar sequence
                   The updated data dictionary adds/updates the following:
            DATA: array of numpy arrays (2D), E2DS data from all fibers in
                  all input exposures.
            BASENAME, string, basename for base FITS file
            HDR: dictionary, header from base FITS file
            CDR: dictionary, header comments from base FITS file
            STOKES: string, stokes parameter detected in sequence
            NEXPOSURES: int, number of exposures in polar sequence
    """

    func_name = __NAME__ + '.load_data()'
    # get constants from p
    stokesparams = p['IC_POLAR_STOKES_PARAMS']
    polarfibers = p['IC_POLAR_FIBERS']

    # First identify which stokes parameter is used in the input data
    stokes_detected = []
    # loop around filenames in polardict
    for filename in polardict.keys():
        # get this entry
        entry = polardict[filename]
        # condition 1: stokes parameter undefined
        cond1 = entry['stokes'].upper() == 'UNDEF'
        # condition 2: stokes parameter in defined parameters
        cond2 = entry['stokes'].upper() in stokesparams
        # condition 3: stokes parameter not already detected
        cond3 = entry['stokes'].upper() not in stokes_detected
        # if (cond1 or cond2) and cond3 append to detected list
        if (cond1 or cond2) and cond3:
            stokes_detected.append(entry['stokes'].upper())
    # if more than one stokes parameter is identified then exit program
    if len(stokes_detected) == 0:
        stokes_detected.append('UNDEF')
    elif len(stokes_detected) > 1:
        wmsg = ('Identified more than one stokes parameter in the input '
                'data... exiting')
        print('error', wmsg)

    # set all possible combinations of fiber type and exposure number
    four_exposure_set = []
    for fiber in polarfibers:
        for exposure in range(1, 5):
            keystr = '{0}_{1}'.format(fiber, exposure)
            four_exposure_set.append(keystr)

    # detect all input combinations of fiber type and exposure number
    four_exposures_detected = []
    loc['RAWFLUXDATA'], loc['RAWFLUXERRDATA'], loc['RAWBLAZEDATA'] = {}, {}, {}
    loc['FLUXDATA'], loc['FLUXERRDATA'] = {}, {}
    loc['WAVEDATA'], loc['BLAZEDATA'] = {}, {}
    loc['TELLURICDATA'] = {}
    
    # loop around the filenames in polardict
    for filename in polardict.keys():
        
        # load SPIRou spectrum
        hdu = fits.open(filename)
        hdr = hdu[0].header
        
        # get this entry
        entry = polardict[filename]
        # get exposure value
        exposure = entry['exposure']
        
        # save basename, wavelength, and object name for 1st exposure:
        if (exposure == 1) :
            loc['BASENAME'] = entry['basename']
            waveAB = deepcopy(hdu["WaveAB"].data)
            if p['IC_POLAR_BERV_CORRECT'] :
                rv_corr = 1.0 + (entry['BERV'] - entry['SOURCE_RV']) / (constants.c / 1000.)
                waveAB *= rv_corr
            loc['WAVE'] = waveAB
            loc['OBJECT'] = hdr['OBJECT']
            loc['HEADER0'] = hdu[0].header
            loc['HEADER1'] = hdu[1].header

        for fiber in polarfibers:
            # set fiber+exposure key string
            keystr = '{0}_{1}'.format(fiber, exposure)
            
            # set flux key for given fiber
            flux_key = "Flux{0}".format(fiber)
            # set wave key for given fiber
            wave_key = "Wave{0}".format(fiber)
            # set blaze key for given fiber
            blaze_key = "Blaze{0}".format(fiber)

            # get flux data
            flux_data = hdu[flux_key].data
            # get normalized blaze data
            blaze_data = hdu[blaze_key].data / np.nanmax(hdu[blaze_key].data)
            # get wavelength data
            wave_data = hdu[wave_key].data
            
            # apply BERV correction if requested
            if p['IC_POLAR_BERV_CORRECT'] :
                rv_corr = 1.0 + (entry['BERV'] - entry['SOURCE_RV']) / (constants.c / 1000.)
                wave_data *= rv_corr

            # store wavelength and blaze vectors
            loc['WAVEDATA'][keystr], loc['RAWBLAZEDATA'][keystr] = wave_data, blaze_data

            # calculate flux errors assuming Poisson noise only
            fluxerr_data = np.zeros_like(flux_data)
            for o in range(len(fluxerr_data)) :
                fluxerr_data[o] = np.sqrt(flux_data[o])

            # save raw flux data and errors
            loc['RAWFLUXDATA'][keystr] = deepcopy(flux_data / blaze_data)
            loc['RAWFLUXERRDATA'][keystr] = deepcopy(fluxerr_data / blaze_data)

            # get shape of flux data
            data_shape = flux_data.shape

            # initialize output arrays to nan
            loc['FLUXDATA'][keystr] = np.empty(data_shape) * np.nan
            loc['FLUXERRDATA'][keystr] = np.empty(data_shape) * np.nan
            loc['BLAZEDATA'][keystr] = np.empty(data_shape) * np.nan
            loc['TELLURICDATA'][keystr] = np.empty(data_shape) * np.nan

            # remove tellurics if possible and if 'IC_POLAR_REMOVE_TELLURICS' parameter is set to "True"
            if entry["TELLURIC_REDUC_FILENAME"] != "" and p['IC_POLAR_REMOVE_TELLURICS'] :
                telluric_spectrum = load_spirou_AB_efits_spectrum(entry["TELLURIC_REDUC_FILENAME"], nan_pos_filter=False)['Recon']
        
            for order_num in range(len(wave_data)) :
                
                clean = ~np.isnan(flux_data[order_num])
                
                if len(wave_data[order_num][clean]) :
                    # interpolate flux data to match wavelength grid of first exposure
                    tck = interpolate.splrep(wave_data[order_num][clean], flux_data[order_num][clean], s=0)
                
                    # interpolate blaze data to match wavelength grid of first exposure
                    btck = interpolate.splrep(wave_data[order_num][clean], blaze_data[order_num][clean], s=0)

                    wlmask = loc['WAVE'][order_num] > wave_data[order_num][clean][0]
                    wlmask &= loc['WAVE'][order_num] < wave_data[order_num][clean][-1]
                
                    loc['BLAZEDATA'][keystr][order_num][wlmask] = interpolate.splev(loc['WAVE'][order_num][wlmask], btck, der=0)
                
                    loc['FLUXDATA'][keystr][order_num][wlmask] = interpolate.splev(loc['WAVE'][order_num][wlmask], tck, der=0) / loc['BLAZEDATA'][keystr][order_num][wlmask]
                    loc['FLUXERRDATA'][keystr][order_num][wlmask] = np.sqrt(loc['FLUXDATA'][keystr][order_num][wlmask] / loc['BLAZEDATA'][keystr][order_num][wlmask] )
                
                    # remove tellurics if possible and if 'IC_POLAR_REMOVE_TELLURICS' parameter is set to "True"
                    if entry["TELLURIC_REDUC_FILENAME"] != "" and p['IC_POLAR_REMOVE_TELLURICS'] :
                        
                        # clean telluric nans
                        clean &= ~np.isnan(telluric_spectrum[order_num])
                        
                        if len(wave_data[order_num][clean]) :
                            # interpolate telluric data
                            ttck = interpolate.splrep(wave_data[order_num][clean], telluric_spectrum[order_num][clean], s=0)
                            
                            loc['TELLURICDATA'][keystr][order_num][clean] = interpolate.splev(loc['WAVE'][order_num][clean], ttck, der=0)
                        
                        # divide spectrum by telluric transmission spectrum
                        loc['FLUXDATA'][keystr][order_num] /= loc['TELLURICDATA'][keystr][order_num]
                        loc['FLUXERRDATA'][keystr][order_num] /= loc['TELLURICDATA'][keystr][order_num]

            # add to four exposure set if correct type
            cond1 = keystr in four_exposure_set
            cond2 = keystr not in four_exposures_detected
            
            if cond1 and cond2:
                four_exposures_detected.append(keystr)

    # initialize number of exposures to zero
    n_exposures = 0
    # now find out whether there is enough exposures
    # first test the 4-exposure set
    if len(four_exposures_detected) == 8:
        n_exposures = 4
    else:
        wmsg = ('Number of exposures in input data is not sufficient'
                ' for polarimetry calculations... exiting')
        print('error', wmsg)

    # set stokes parameters defined
    loc['STOKES'] = stokes_detected[0]
    # set the number of exposures detected
    loc['NEXPOSURES'] = n_exposures

    # calculate time related quantities
    loc = calculate_polar_times(p, polardict, loc)

    # return loc
    return p, loc


def calculate_polar_times(p, polardict, loc) :
    """
        Function to calculate time related quantities of polar product
        
        :param p: parameter dictionary, ParamDict containing constants
        
        :param loc: parameter dictionary, ParamDict containing data
        
        :param polardict: dictionary, ParamDict containing information on the
        input data
    """
    
    mjd_first, mjd_last = 0.0, 0.0
    meanbjd, tot_exptime = 0.0, 0.0
    bjd_first, bjd_last, exptime_last = 0.0, 0.0, 0.0
    berv_first, berv_last = 0.0, 0.0
    bervmaxs = []
    
    # loop over files in polar sequence
    for filename in polardict.keys():
        # get expnum
        expnum = polardict[filename]['exposure']
        # get header
        hdr0 = fits.getheader(filename)
        hdr1 = fits.getheader(filename,1)
        # calcualte total exposure time
        tot_exptime += float(hdr0['EXPTIME'])
        # get values for BJDCEN calculation
        if expnum == 1:
            mjd_first = float(hdr0['MJDATE'])
            bjd_first = float(hdr1['BJD'])
            berv_first = float(hdr1['BERV'])
        elif expnum == loc['NEXPOSURES']:
            mjd_last = float(hdr0['MJDATE'])
            bjd_last = float(hdr1['BJD'])
            berv_last = float(hdr1['BERV'])
            exptime_last = float(hdr0['EXPTIME'])
        meanbjd += float(hdr1['BJD'])
        # append BERVMAX value of each exposure
        bervmaxs.append(float(hdr1['BERVMAX']))

    # add elapsed time parameter keyword to header
    elapsed_time = (bjd_last - bjd_first) * 86400. + exptime_last
    loc['ELAPSED_TIME'] = elapsed_time

    # calculate MJD at center of polarimetric sequence
    mjdcen = mjd_first + (mjd_last - mjd_first + exptime_last/86400.)/2.0
    loc['MJDCEN'] = mjdcen

    # calculate BJD at center of polarimetric sequence
    bjdcen = bjd_first + (bjd_last - bjd_first + exptime_last/86400.)/2.0
    loc['BJDCEN'] = bjdcen

    # calculate BERV at center by linear interpolation
    berv_slope = (berv_last - berv_first) / (bjd_last - bjd_first)
    berv_intercept = berv_first - berv_slope * bjd_first
    loc['BERVCEN'] = berv_intercept + berv_slope * bjdcen

    # calculate maximum bervmax
    bervmax = np.max(bervmaxs)
    loc['BERVMAX'] = bervmax

    # add mean BJD
    meanbjd = meanbjd / loc['NEXPOSURES']
    loc['MEANBJD'] = meanbjd
    
    return loc


def calculate_polarimetry(p, loc):
    """
    Function to call functions to calculate polarimetry either using
    the Ratio or Difference methods.
        
    :param p: parameter dictionary, ParamDict containing constants
        Must contain at least:
            LOG_OPT: string, option for logging
            IC_POLAR_METHOD: string, to define polar method "Ratio" or
                             "Difference"

    :param loc: parameter dictionary, ParamDict containing data
        
    :return polarfunc: function, either polarimetry_diff_method(p, loc)
                       or polarimetry_ratio_method(p, loc)
    """

    # get parameters from p
    method = p['IC_POLAR_METHOD']
    # decide which method to use
    if method == 'Difference':
        return polarimetry_diff_method(p, loc)
    elif method == 'Ratio':
        return polarimetry_ratio_method(p, loc)
    else:
        emsg = 'Method="{0}" not valid for polarimetry calculation'
        print('error', emsg.format(method))
        return 1


def calculate_stokes_i(p, loc):
    """
    Function to calculate the Stokes I polarization

    :param p: parameter dictionary, ParamDict containing constants
        Must contain at least:
            LOG_OPT: string, option for logging

    :param loc: parameter dictionary, ParamDict containing data
        Must contain at least:
            DATA: array of numpy arrays (2D), E2DS data from all fibers in
                  all input exposures.
            NEXPOSURES: int, number of exposures in polar sequence

    :return loc: parameter dictionary, the updated parameter dictionary
        Adds/updates the following:
            STOKESI: numpy array (2D), the Stokes I parameters, same shape as
                     DATA
            STOKESIERR: numpy array (2D), the Stokes I error parameters, same
                        shape as DATA
    """
    func_name = __NAME__ + '.calculate_stokes_I()'
    name = 'CalculateStokesI'
    # log start of Stokes I calculations
    wmsg = 'Running function {0} to calculate Stokes I total flux'
    print('info', wmsg.format(name))
    # get parameters from loc
    if p['IC_POLAR_INTERPOLATE_FLUX'] :
        data, errdata = loc['FLUXDATA'], loc['FLUXERRDATA']
    else :
        data, errdata = loc['RAWFLUXDATA'], loc['RAWFLUXERRDATA']
    nexp = float(loc['NEXPOSURES'])
    # ---------------------------------------------------------------------
    # set up storage
    # ---------------------------------------------------------------------
    # store Stokes I variables in loc
    data_shape = loc['FLUXDATA']['A_1'].shape
    # initialize arrays to zeroes
    loc['STOKESI'] = np.zeros(data_shape)
    loc['STOKESIERR'] = np.zeros(data_shape)

    flux, var = [], []
    for i in range(1, int(nexp) + 1):
        # Calculate sum of fluxes from fibers A and B
        flux_ab = data['A_{0}'.format(i)] + data['B_{0}'.format(i)]
        # Save A+B flux for each exposure
        flux.append(flux_ab)

        # Calculate the variances for fiber A+B -> varA+B = sigA * sigA + sigB * sigB
        var_ab = errdata['A_{0}'.format(i)] * errdata['A_{0}'.format(i)] + errdata['B_{0}'.format(i)] * errdata['B_{0}'.format(i)]
        # Save varAB = sigA^2 + sigB^2, ignoring cross-correlated terms
        var.append(var_ab)

    # Sum fluxes and variances from different exposures
    for i in range(len(flux)):
        loc['STOKESI'] += flux[i]
        loc['STOKESIERR'] += var[i]

    # Calcualte errors -> sigma = sqrt(variance)
    loc['STOKESIERR'] = np.sqrt(loc['STOKESIERR'])

    # log end of Stokes I intensity calculations
    wmsg = 'Routine {0} run successfully'
    print('info', wmsg.format(name))
    # return loc
    return loc


def polarimetry_diff_method(p, loc):
    """
    Function to calculate polarimetry using the difference method as described
    in the paper:
        Bagnulo et al., PASP, Volume 121, Issue 883, pp. 993 (2009)
        
    :param p: parameter dictionary, ParamDict containing constants
        Must contain at least:
            p['LOG_OPT']: string, option for logging
        
    :param loc: parameter dictionary, ParamDict containing data
        Must contain at least:
            loc['RAWFLUXDATA']: numpy array (2D) containing the e2ds flux data for all
                                exposures {1,..,NEXPOSURES}, and for all fibers {A,B}
            loc['RAWFLUXERRDATA']: numpy array (2D) containing the e2ds flux error data for all
                                   exposures {1,..,NEXPOSURES}, and for all fibers {A,B}
            loc['NEXPOSURES']: number of polarimetry exposures
        
    :return loc: parameter dictionary, the updated parameter dictionary
        Adds/updates the following:
            loc['POL']: numpy array (2D), degree of polarization data, which
                        should be the same shape as E2DS files, i.e, 
                        loc[DATA][FIBER_EXP]
            loc['POLERR']: numpy array (2D), errors of degree of polarization,
                           same shape as loc['POL']
            loc['NULL1']: numpy array (2D), 1st null polarization, same 
                          shape as loc['POL']
            loc['NULL2']: numpy array (2D), 2nd null polarization, same 
                          shape as loc['POL']
    """

    func_name = __NAME__ + '.polarimetry_diff_method()'
    name = 'polarimetryDiffMethod'
    # log start of polarimetry calculations
    wmsg = 'Running function {0} to calculate polarization'
    print('info', wmsg.format(name))
    # get parameters from loc
    if p['IC_POLAR_INTERPOLATE_FLUX'] :
        data, errdata = loc['FLUXDATA'], loc['FLUXERRDATA']
    else :
        data, errdata = loc['RAWFLUXDATA'], loc['RAWFLUXERRDATA']
    nexp = float(loc['NEXPOSURES'])
    # ---------------------------------------------------------------------
    # set up storage
    # ---------------------------------------------------------------------
    # store polarimetry variables in loc
    data_shape = loc['RAWFLUXDATA']['A_1'].shape
    # initialize arrays to zeroes
    loc['POL'] = np.zeros(data_shape)
    loc['POLERR'] = np.zeros(data_shape)
    loc['NULL1'] = np.zeros(data_shape)
    loc['NULL2'] = np.zeros(data_shape)

    gg, gvar = [], []
    for i in range(1, int(nexp) + 1):
        # ---------------------------------------------------------------------
        # STEP 1 - calculate the quantity Gn (Eq #12-14 on page 997 of
        #          Bagnulo et al. 2009), n being the pair of exposures
        # ---------------------------------------------------------------------
        part1 = data['A_{0}'.format(i)] - data['B_{0}'.format(i)]
        part2 = data['A_{0}'.format(i)] + data['B_{0}'.format(i)]
        gg.append(part1 / part2)

        # Calculate the variances for fiber A and B:
        a_var = errdata['A_{0}'.format(i)] * errdata['A_{0}'.format(i)]
        b_var = errdata['B_{0}'.format(i)] * errdata['B_{0}'.format(i)]

        # ---------------------------------------------------------------------
        # STEP 2 - calculate the quantity g_n^2 (Eq #A4 on page 1013 of
        #          Bagnulo et al. 2009), n being the pair of exposures
        # ---------------------------------------------------------------------
        nomin = 2.0 * data['A_{0}'.format(i)] * data['B_{0}'.format(i)]
        denom = (data['A_{0}'.format(i)] + data['B_{0}'.format(i)]) ** 2.0
        factor1 = (nomin / denom) ** 2.0
        a_var_part = a_var / (data['A_{0}'.format(i)] * data['A_{0}'.format(i)])
        b_var_part = b_var / (data['B_{0}'.format(i)] * data['B_{0}'.format(i)])
        gvar.append(factor1 * (a_var_part + b_var_part))

    # if we have 4 exposures
    if nexp == 4:
        # -----------------------------------------------------------------
        # STEP 3 - calculate the quantity Dm (Eq #18 on page 997 of
        #          Bagnulo et al. 2009 paper) and the quantity Dms with
        #          exposures 2 and 4 swapped, m being the pair of exposures
        #          Ps. Notice that SPIRou design is such that the angles of
        #          the exposures that correspond to different angles of the
        #          retarder are obtained in the order (1)->(2)->(4)->(3),
        #          which explains the swap between G[3] and G[2].
        # -----------------------------------------------------------------
        d1, d2 = gg[0] - gg[1], gg[3] - gg[2]
        d1s, d2s = gg[0] - gg[2], gg[3] - gg[1]
        # -----------------------------------------------------------------
        # STEP 4 - calculate the degree of polarization for Stokes
        #          parameter (Eq #19 on page 997 of Bagnulo et al. 2009)
        # -----------------------------------------------------------------
        loc['POL'] = (d1 + d2) / nexp
        # -----------------------------------------------------------------
        # STEP 5 - calculate the first NULL spectrum
        #          (Eq #20 on page 997 of Bagnulo et al. 2009)
        # -----------------------------------------------------------------
        loc['NULL1'] = (d1 - d2) / nexp
        # -----------------------------------------------------------------
        # STEP 6 - calculate the second NULL spectrum
        #          (Eq #20 on page 997 of Bagnulo et al. 2009)
        #          with exposure 2 and 4 swapped
        # -----------------------------------------------------------------
        loc['NULL2'] = (d1s - d2s) / nexp
        # -----------------------------------------------------------------
        # STEP 7 - calculate the polarimetry error
        #          (Eq #A3 on page 1013 of Bagnulo et al. 2009)
        # -----------------------------------------------------------------
        sum_of_gvar = gvar[0] + gvar[1] + gvar[2] + gvar[3]
        loc['POLERR'] = np.sqrt(sum_of_gvar / (nexp ** 2.0))

    # else if we have 2 exposures
    elif nexp == 2:
        # -----------------------------------------------------------------
        # STEP 3 - calculate the quantity Dm
        #          (Eq #18 on page 997 of Bagnulo et al. 2009) and
        #          the quantity Dms with exposure 2 and 4 swapped,
        #          m being the pair of exposures
        # -----------------------------------------------------------------
        d1 = gg[0] - gg[1]
        # -----------------------------------------------------------------
        # STEP 4 - calculate the degree of polarization
        #          (Eq #19 on page 997 of Bagnulo et al. 2009)
        # -----------------------------------------------------------------
        loc['POL'] = d1 / nexp
        # -----------------------------------------------------------------
        # STEP 5 - calculate the polarimetry error
        #          (Eq #A3 on page 1013 of Bagnulo et al. 2009)
        # -----------------------------------------------------------------
        sum_of_gvar = gvar[0] + gvar[1]
        loc['POLERR'] = np.sqrt(sum_of_gvar / (nexp ** 2.0))

    # else we have insufficient data (should not get here)
    else:
        wmsg = ('Number of exposures in input data is not sufficient'
                ' for polarimetry calculations... exiting')
        print('error', wmsg)

    # set the method
    loc['METHOD'] = 'Difference'

    # log end of polarimetry calculations
    wmsg = 'Routine {0} run successfully'
    print('info', wmsg.format(name))
    # return loc
    return loc


def polarimetry_ratio_method(p, loc):
    """
    Function to calculate polarimetry using the ratio method as described
    in the paper:
        Bagnulo et al., PASP, Volume 121, Issue 883, pp. 993 (2009)
        
    :param p: parameter dictionary, ParamDict containing constants
        Must contain at least:
            p['LOG_OPT']: string, option for logging
        
    :param loc: parameter dictionary, ParamDict containing data
        Must contain at least:
        loc['RAWFLUXDATA']: numpy array (2D) containing the e2ds flux data for all
                            exposures {1,..,NEXPOSURES}, and for all fibers {A,B}
        loc['RAWFLUXERRDATA']: numpy array (2D) containing the e2ds flux error data for all
                             exposures {1,..,NEXPOSURES}, and for all fibers {A,B}
        loc['NEXPOSURES']: number of polarimetry exposures
        
    :return loc: parameter dictionary, the updated parameter dictionary
        Adds/updates the following:
            loc['POL']: numpy array (2D), degree of polarization data, which
                        should be the same shape as E2DS files, i.e, 
                        loc[DATA][FIBER_EXP]
            loc['POLERR']: numpy array (2D), errors of degree of polarization,
                           same shape as loc['POL']
            loc['NULL1']: numpy array (2D), 1st null polarization, same 
                          shape as loc['POL']
            loc['NULL2']: numpy array (2D), 2nd null polarization, same 
                          shape as loc['POL']
    """
    func_name = __NAME__ + '.polarimetry_ratio_method()'
    name = 'polarimetryRatioMethod'

    # log start of polarimetry calculations
    wmsg = 'Running function {0} to calculate polarization'
    print('info', wmsg.format(name))
    # get parameters from loc
    if p['IC_POLAR_INTERPOLATE_FLUX'] :
        data, errdata = loc['FLUXDATA'], loc['FLUXERRDATA']
    else :
        data, errdata = loc['RAWFLUXDATA'], loc['RAWFLUXERRDATA']
    
    nexp = float(loc['NEXPOSURES'])
    # ---------------------------------------------------------------------
    # set up storage
    # ---------------------------------------------------------------------
    # store polarimetry variables in loc
    data_shape = loc['RAWFLUXDATA']['A_1'].shape
    # initialize arrays to zeroes
    loc['POL'] = np.zeros(data_shape)
    loc['POLERR'] = np.zeros(data_shape)
    loc['NULL1'] = np.zeros(data_shape)
    loc['NULL2'] = np.zeros(data_shape)
    
    flux_ratio, var_term = [], []

    # Ignore numpy warnings to avoid warning message: "RuntimeWarning: invalid
    # value encountered in power ..."
    np.warnings.filterwarnings('ignore')

    for i in range(1, int(nexp) + 1):
        # ---------------------------------------------------------------------
        # STEP 1 - calculate ratio of beams for each exposure
        #          (Eq #12 on page 997 of Bagnulo et al. 2009 )
        # ---------------------------------------------------------------------
        part1 = data['A_{0}'.format(i)]
        part2 = data['B_{0}'.format(i)]
        flux_ratio.append(part1 / part2)

        # Calculate the variances for fiber A and B:
        a_var = errdata['A_{0}'.format(i)] * errdata['A_{0}'.format(i)]
        b_var = errdata['B_{0}'.format(i)] * errdata['B_{0}'.format(i)]

        # ---------------------------------------------------------------------
        # STEP 2 - calculate the error quantities for Eq #A10 on page 1014 of
        #          Bagnulo et al. 2009
        # ---------------------------------------------------------------------
        var_term_part1 = a_var / (data['A_{0}'.format(i)] * data['A_{0}'.format(i)])
        var_term_part2 = b_var / (data['B_{0}'.format(i)] * data['B_{0}'.format(i)])
        var_term.append(var_term_part1 + var_term_part2)

    # if we have 4 exposures
    if nexp == 4:
        # -----------------------------------------------------------------
        # STEP 3 - calculate the quantity Rm
        #          (Eq #23 on page 998 of Bagnulo et al. 2009) and
        #          the quantity Rms with exposure 2 and 4 swapped,
        #          m being the pair of exposures
        #          Ps. Notice that SPIRou design is such that the angles of
        #          the exposures that correspond to different angles of the
        #          retarder are obtained in the order (1)->(2)->(4)->(3),which
        #          explains the swap between flux_ratio[3] and flux_ratio[2].
        # -----------------------------------------------------------------
        r1, r2 = flux_ratio[0] / flux_ratio[1], flux_ratio[3] / flux_ratio[2]
        r1s, r2s = flux_ratio[0] / flux_ratio[2], flux_ratio[3] / flux_ratio[1]
        # -----------------------------------------------------------------
        # STEP 4 - calculate the quantity R
        #          (Part of Eq #24 on page 998 of Bagnulo et al. 2009)
        # -----------------------------------------------------------------
        rr = (r1 * r2) ** (1.0 / nexp)
        # -----------------------------------------------------------------
        # STEP 5 - calculate the degree of polarization
        #          (Eq #24 on page 998 of Bagnulo et al. 2009)
        # -----------------------------------------------------------------
        loc['POL'] = (rr - 1.0) / (rr + 1.0)
        # -----------------------------------------------------------------
        # STEP 6 - calculate the quantity RN1
        #          (Part of Eq #25-26 on page 998 of Bagnulo et al. 2009)
        # -----------------------------------------------------------------
        rn1 = (r1 / r2) ** (1.0 / nexp)
        # -----------------------------------------------------------------
        # STEP 7 - calculate the first NULL spectrum
        #          (Eq #25-26 on page 998 of Bagnulo et al. 2009)
        # -----------------------------------------------------------------
        loc['NULL1'] = (rn1 - 1.0) / (rn1 + 1.0)
        # -----------------------------------------------------------------
        # STEP 8 - calculate the quantity RN2
        #          (Part of Eq #25-26 on page 998 of Bagnulo et al. 2009),
        #          with exposure 2 and 4 swapped
        # -----------------------------------------------------------------
        rn2 = (r1s / r2s) ** (1.0 / nexp)
        # -----------------------------------------------------------------
        # STEP 9 - calculate the second NULL spectrum
        #          (Eq #25-26 on page 998 of Bagnulo et al. 2009),
        #          with exposure 2 and 4 swapped
        # -----------------------------------------------------------------
        loc['NULL2'] = (rn2 - 1.0) / (rn2 + 1.0)
        # -----------------------------------------------------------------
        # STEP 10 - calculate the polarimetry error (Eq #A10 on page 1014
        #           of Bagnulo et al. 2009)
        # -----------------------------------------------------------------
        numer_part1 = (r1 * r2) ** (1.0 / 2.0)
        denom_part1 = ((r1 * r2) ** (1.0 / 4.0) + 1.0) ** 4.0
        part1 = numer_part1 / (denom_part1 * 4.0)
        sumvar = var_term[0] + var_term[1] + var_term[2] + var_term[3]
        loc['POLERR'] = np.sqrt(part1 * sumvar)

    # else if we have 2 exposures
    elif nexp == 2:
        # -----------------------------------------------------------------
        # STEP 3 - calculate the quantity Rm
        #          (Eq #23 on page 998 of Bagnulo et al. 2009) and
        #          the quantity Rms with exposure 2 and 4 swapped,
        #          m being the pair of exposures
        # -----------------------------------------------------------------
        r1 = flux_ratio[0] / flux_ratio[1]

        # -----------------------------------------------------------------
        # STEP 4 - calculate the quantity R
        #          (Part of Eq #24 on page 998 of Bagnulo et al. 2009)
        # -----------------------------------------------------------------
        rr = r1 ** (1.0 / nexp)

        # -----------------------------------------------------------------
        # STEP 5 - calculate the degree of polarization
        #          (Eq #24 on page 998 of Bagnulo et al. 2009)
        # -----------------------------------------------------------------
        loc['POL'] = (rr - 1.0) / (rr + 1.0)
        # -----------------------------------------------------------------
        # STEP 6 - calculate the polarimetry error (Eq #A10 on page 1014
        #           of Bagnulo et al. 2009)
        # -----------------------------------------------------------------
        # numer_part1 = R1
        denom_part1 = ((r1 ** 0.5) + 1.0) ** 4.0
        part1 = r1 / denom_part1
        sumvar = var_term[0] + var_term[1]
        loc['POLERR'] = np.sqrt(part1 * sumvar)

    # else we have insufficient data (should not get here)
    else:
        wmsg = ('Number of exposures in input data is not sufficient'
                ' for polarimetry calculations... exiting')
        print('error', wmsg)

    # set the method
    loc['METHOD'] = 'Ratio'
    # log end of polarimetry calculations
    wmsg = 'Routine {0} run successfully'
    print('info', wmsg.format(name))
    # return loc
    return loc



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


#### Function to detect continuum #########
def continuum_polarization(x, y, binsize=200, overlap=100, window=20, mode="median", use_polynomail_fit=True, deg_poly_fit = 3, telluric_bands=[]):
    """
    Function to calculate continuum polarization
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
        
        if i == 0 :
            xbin.append(x[0] - np.abs(x[1] - x[0]))
            # create mask to get rid of NaNs
            localnanmask = np.logical_not(np.isnan(y))
            ybin.append(np.median(y[localnanmask][:binsize]))
        
        if len(xtmp[nanmask]) > 2 :
            # calculate mean x within the bin
            xmean = np.mean(xtmp[nanmask])
        
            # save mean x wihthin bin
            xbin.append(xmean)

            if mode == 'median':
                # save median y of filtered data
                ybin.append(np.median(ytmp[nanmask]))
            elif mode == 'mean':
                # save mean y of filtered data
                ybin.append(np.mean(ytmp[nanmask]))
            else:
                emsg = 'Can not recognize selected mode="{0}"...exiting'
                print('error', emsg.format(mode))
                    
        if i == nbins - 1 :
            xbin.append(x[-1] + np.abs(x[-1] - x[-2]))
            # create mask to get rid of NaNs
            localnanmask = np.logical_not(np.isnan(y))
            ybin.append(np.median(y[localnanmask][-binsize:]))

    # the continuum may be obtained either by polynomial fit or by cubic interpolation
    if use_polynomail_fit :
    # Option to use a polynomial fit
        # Fit polynomial function to sample points
        pfit = np.polyfit(xbin, ybin, deg_poly_fit)
        # Set numpy poly1d objects
        p = np.poly1d(pfit)
        # Evaluate polynomial in the original grid
        cont = p(x)
    else :
    # option to interpolate points applying a cubic spline to the continuum data
        sfit = interp1d(xbin, ybin, kind='cubic')
        # Resample interpolation to the original grid
        cont = sfit(x)

    # return continuum polarization and x and y bins
    return cont, xbin, ybin
    ##-- end of continuum polarization function


def calculate_continuum(p, loc, in_wavelength=True):
    """
    Function to calculate the continuum flux and continuum polarization
    
    :param p: parameter dictionary, ParamDict containing constants
        Must contain at least:
            LOG_OPT: string, option for logging
            IC_POLAR_CONT_BINSIZE: int, number of points in each sample bin
            IC_POLAR_CONT_OVERLAP: int, number of points to overlap before and
                                   after each sample bin
            IC_POLAR_CONT_TELLMASK: list of float pairs, list of telluric bands,
                                    i.e, a list of wavelength ranges ([wl0,wlf])
                                    for telluric absorption
        
    :param loc: parameter dictionary, ParamDict containing data
        Must contain at least:
            WAVE: numpy array (2D), e2ds wavelength data
            POL: numpy array (2D), e2ds degree of polarization data
            POLERR: numpy array (2D), e2ds errors of degree of polarization
            NULL1: numpy array (2D), e2ds 1st null polarization
            NULL2: numpy array (2D), e2ds 2nd null polarization
            STOKESI: numpy array (2D), e2ds Stokes I data
            STOKESIERR: numpy array (2D), e2ds errors of Stokes I
        
    :param in_wavelength: bool, to indicate whether or not there is wave cal

    :return loc: parameter dictionary, the updated parameter dictionary
        Adds/updates the following:
            FLAT_X: numpy array (1D), flatten polarimetric x data
            FLAT_POL: numpy array (1D), flatten polarimetric pol data
            FLAT_POLERR: numpy array (1D), flatten polarimetric pol error data
            FLAT_STOKESI: numpy array (1D), flatten polarimetric stokes I data
            FLAT_STOKESIERR: numpy array (1D), flatten polarimetric stokes I
                             error data
            FLAT_NULL1: numpy array (1D), flatten polarimetric null1 data
            FLAT_NULL2: numpy array (1D), flatten polarimetric null2 data
            CONT_FLUX: numpy array (1D), e2ds continuum flux data
                       interpolated from xbin, ybin points, same shape as FLAT_STOKESI
            CONT_FLUX_XBIN: numpy array (1D), continuum in x flux samples
            CONT_FLUX_YBIN: numpy array (1D), continuum in y flux samples

            CONT_POL: numpy array (1D), e2ds continuum polarization data
                      interpolated from xbin, ybin points, same shape as
                      FLAT_POL
            CONT_POL_XBIN: numpy array (1D), continuum in x polarization samples
            CONT_POL_YBIN: numpy array (1D), continuum in y polarization samples
    """

    func_name = __NAME__ + '.calculate_continuum()'
    # get constants from p
    pol_binsize = p['IC_POLAR_CONT_BINSIZE']
    pol_overlap = p['IC_POLAR_CONT_OVERLAP']
    # get wavelength data if require
    if in_wavelength:
        wldata = loc['WAVE']
    else:
        wldata = np.ones_like(loc['POL'])
    # get the shape of pol
    ydim, xdim = loc['POL'].shape
    # ---------------------------------------------------------------------
    # flatten data (across orders)
    wl, pol, polerr, stokes_i, stokes_ierr = [], [], [], [], []
    null1, null2 = [], []
    # loop around order data
    for order_num in range(ydim):
        if in_wavelength:
            wl = np.append(wl, wldata[order_num])
        else:
            wl = np.append(wl, (order_num * xdim) + np.arange(xdim))
        pol = np.append(pol, loc['POL'][order_num])
        polerr = np.append(polerr, loc['POLERR'][order_num])
        stokes_i = np.append(stokes_i, loc['STOKESI'][order_num])
        stokes_ierr = np.append(stokes_ierr, loc['STOKESIERR'][order_num])
        null1 = np.append(null1, loc['NULL1'][order_num])
        null2 = np.append(null2, loc['NULL2'][order_num])
    # ---------------------------------------------------------------------
    # sort by wavelength (or pixel number)
    sortmask = np.argsort(wl)

    # save back to loc
    loc['FLAT_X'] = wl[sortmask]
    loc['FLAT_POL'] = pol[sortmask]
    loc['FLAT_POLERR'] = polerr[sortmask]
    loc['FLAT_STOKESI'] = stokes_i[sortmask]
    loc['FLAT_STOKESIERR'] = stokes_ierr[sortmask]
    loc['FLAT_NULL1'] = null1[sortmask]
    loc['FLAT_NULL2'] = null2[sortmask]

    # ---------------------------------------------------------------------
    # calculate continuum flux
    contflux, xbin, ybin = continuum(loc['FLAT_X'], loc['FLAT_STOKESI'],
                                    binsize=pol_binsize, overlap=pol_overlap,
                                    window=6, mode="max", use_linear_fit=True,
                                    telluric_bands=p['IC_POLAR_CONT_TELLMASK'])
    # ---------------------------------------------------------------------
    # save continuum data to loc
    loc['CONT_FLUX'] = contflux
    loc['CONT_FLUX_XBIN'] = xbin
    loc['CONT_FLUX_YBIN'] = ybin

    # normalize flux by continuum
    if p['IC_POLAR_NORMALIZE_STOKES_I'] :
        loc['FLAT_STOKESI'] /= loc['CONT_FLUX']
        loc['FLAT_STOKESIERR'] /= loc['CONT_FLUX']

    # ---------------------------------------------------------------------
    # calculate continuum polarization
    contpol, xbinpol, ybinpol = continuum_polarization(loc['FLAT_X'], loc['FLAT_POL'],
                                                binsize=pol_binsize, overlap=pol_overlap,
                                                mode="median",
                                                use_polynomail_fit=p['IC_POLAR_CONT_POLYNOMIAL_FIT'], deg_poly_fit = p['IC_POLAR_CONT_DEG_POLYNOMIAL'],
                                                telluric_bands=p['IC_POLAR_CONT_TELLMASK'])
    # ---------------------------------------------------------------------


    #plt.plot(loc['FLAT_X'], loc['FLAT_POL'],'.')
    #plt.plot(xbinpol, ybinpol,'o')
    #plt.plot(loc['FLAT_X'], contpol, '-')
    #plt.show()

    # save continuum data to loc
    loc['CONT_POL'] = contpol
    loc['CONT_POL_XBIN'] = xbinpol
    loc['CONT_POL_YBIN'] = ybinpol

    # remove continuum polarization
    if p['IC_POLAR_REMOVE_CONTINUUM'] :
        loc['FLAT_POL'] -= loc['CONT_POL']

    # return loc
    return loc


def remove_continuum_polarization(loc):
    """
        Function to remove the continuum polarization
        
        :param loc: parameter dictionary, ParamDict containing data
        Must contain at least:
        WAVE: numpy array (2D), e2ds wavelength data
        POL: numpy array (2D), e2ds degree of polarization data
        POLERR: numpy array (2D), e2ds errors of degree of polarization
        FLAT_X: numpy array (1D), flatten polarimetric x data
        CONT_POL: numpy array (1D), e2ds continuum polarization data

        :return loc: parameter dictionary, the updated parameter dictionary
        Adds/updates the following:
        POL: numpy array (2D), e2ds degree of polarization data
        ORDER_CONT_POL: numpy array (2D), e2ds degree of continuum polarization data
        """

    func_name = __NAME__ + '.remove_continuum()'
    
    # get the shape of pol
    ydim, xdim = loc['POL'].shape
    
    # initialize continuum empty array
    loc['ORDER_CONT_POL'] = np.empty(loc['POL'].shape) * np.nan
    
    # ---------------------------------------------------------------------
    # interpolate and remove continuum (across orders)
    # loop around order data
    for order_num in range(ydim):
        
        # get wavelengths for current order
        wl = loc['WAVE'][order_num]
        
        # get wavelength at edges of order
        wl0, wlf = wl[0], wl[-1]
        
        # get polarimetry for current order
        pol = loc['POL'][order_num]

        # create mask to get only continuum data within wavelength range
        wlmask = np.where(np.logical_and(loc['FLAT_X'] >= wl0,
                                         loc['FLAT_X'] <= wlf))

        # get continuum data within order range
        wl_cont = loc['FLAT_X'][wlmask]
        pol_cont = loc['CONT_POL'][wlmask]
        
        # interpolate points applying a cubic spline to the continuum data
        f = interp1d(wl_cont, pol_cont, kind='cubic')
    
        # create continuum vector at same wavelength sampling as polar data
        continuum = f(wl)
        
        # save continuum with the same shape as input pol
        loc['ORDER_CONT_POL'][order_num] = continuum

        # remove continuum from data
        loc['POL'][order_num] = pol - continuum

    return loc

def normalize_stokes_i(loc) :
    """
        Function to normalize Stokes I by the continuum flux
        
        :param loc: parameter dictionary, ParamDict containing data
        Must contain at least:
        WAVE: numpy array (2D), e2ds wavelength data
        STOKESI: numpy array (2D), e2ds degree of polarization data
        POLERR: numpy array (2D), e2ds errors of degree of polarization
        FLAT_X: numpy array (1D), flatten polarimetric x data
        CONT_POL: numpy array (1D), e2ds continuum polarization data

        :return loc: parameter dictionary, the updated parameter dictionary
        Adds/updates the following:
        STOKESI: numpy array (2D), e2ds Stokes I data
        STOKESIERR: numpy array (2D), e2ds Stokes I error data
        ORDER_CONT_FLUX: numpy array (2D), e2ds flux continuum data
        """

    func_name = __NAME__ + '.remove_continuum()'
    
    # get the shape of pol
    ydim, xdim = loc['STOKESI'].shape
    
    # initialize continuum empty array
    loc['ORDER_CONT_FLUX'] = np.empty(loc['STOKESI'].shape) * np.nan

    # ---------------------------------------------------------------------
    # interpolate and remove continuum (across orders)
    # loop around order data
    for order_num in range(ydim):
        
        # get wavelengths for current order
        wl = loc['WAVE'][order_num]
        
        # get wavelength at edges of order
        wl0, wlf = wl[0], wl[-1]
        
        # get polarimetry for current order
        flux = loc['STOKESI'][order_num]
        fluxerr = loc['STOKESIERR'][order_num]

        # create mask to get only continuum data within wavelength range
        wlmask = np.where(np.logical_and(loc['FLAT_X'] >= wl0,
                                         loc['FLAT_X'] <= wlf))

        # get continuum data within order range
        wl_cont = loc['FLAT_X'][wlmask]
        flux_cont = loc['CONT_FLUX'][wlmask]
        
        # interpolate points applying a cubic spline to the continuum data
        f = interp1d(wl_cont, flux_cont, kind='cubic')
    
        # create continuum vector at same wavelength sampling as polar data
        continuum = f(wl)
        
        # save continuum with the same shape as input pol
        loc['ORDER_CONT_FLUX'][order_num] = continuum
        
        # normalize stokes I by the continuum
        loc['STOKESI'][order_num] = flux / continuum
        # normalize stokes I by the continuum
        loc['STOKESIERR'][order_num] = fluxerr / continuum

    return loc


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


# TODO: Need a better fix for this
def attempt_tcl_error_fix():
    plt.switch_backend('agg')


def end_plotting(p, plot_name):
    """
    End plotting properly (depending on DRS_PLOT and interactive mode)

    :param p: ParamDict, the constants parameter dictionary
    :param plot_name:
    :return:
    """
    
    """
    if p['DRS_PLOT'] == 2:
        # get plotting figure names (as a list for multiple formats)
        snames = define_save_name(p, plot_name)
        # loop around formats
        for sname in snames:
            # log plot saving
            wmsg = 'Saving plot to {0}'
            print('', wmsg.format(sname))
            # save figure
            plt.savefig(sname)
        # close figure cleanly
        plt.close()
        # do not contibue with interactive tests --> return here
        return 0
    """
    # turn off interactive plotting
    if not plt.isinteractive():
        plt.show()
        plt.close()
    else:
        pass


# =============================================================================
# Polarimetry plotting functions
# =============================================================================
def polar_continuum_plot(p, loc, in_wavelengths=True):
    plot_name = 'polar_continuum_plot'
    # get data from loc
    wl, pol = loc['FLAT_X'], 100.0 * (loc['FLAT_POL'] + loc['CONT_POL'])
    contpol = 100.0 * loc['CONT_POL']
    contxbin, contybin = np.array(loc['CONT_POL_XBIN']), np.array(loc['CONT_POL_YBIN'])
    contybin = 100. * contybin
    stokes = loc['STOKES']
    method, nexp = loc['METHOD'], loc['NEXPOSURES']

    # ---------------------------------------------------------------------
    # set up fig
    fig, frame = setup_figure(p)
    # ---------------------------------------------------------------------
    # set up labels
    if in_wavelengths:
        xlabel = 'wavelength (nm)'
    else:
        xlabel = 'order number + col (pixel)'
    ylabel = 'Degree of polarization for Stokes {0} (%)'.format(stokes)
    # set up title
    title = 'Polarimetry: Stokes {0}, Method={1}, for {2} exposures'
    titleargs = [stokes, method, nexp]
    # ---------------------------------------------------------------------
    # plot polarimetry data
    frame.plot(wl, pol, linestyle='None', marker='.',
               label='Degree of Polarization')
    # plot continuum sample points
    frame.plot(contxbin, contybin, linestyle='None', marker='o',
               label='Continuum Samples')
    # plot continuum fit
    frame.plot(wl, contpol, label='Continuum Polarization')
    # ---------------------------------------------------------------------
    # set title and labels
    frame.set(title=title.format(*titleargs), xlabel=xlabel, ylabel=ylabel)
    # ---------------------------------------------------------------------
    # plot legend
    frame.legend(loc=0)
    # ---------------------------------------------------------------------
    # end plotting function properly
    end_plotting(p, plot_name)


def polar_result_plot(p, loc, in_wavelengths=True):
    plot_name = 'polar_result_plot'
    # get data from loc
    wl, pol = loc['FLAT_X'], 100.0 * loc['FLAT_POL']
    null1, null2 = 100.0 * loc['FLAT_NULL1'], 100.0 * loc['FLAT_NULL2']
    stokes = loc['STOKES']
    method, nexp = loc['METHOD'], loc['NEXPOSURES']
    # ---------------------------------------------------------------------
    # set up fig
    fig, frame = setup_figure(p)
    # ---------------------------------------------------------------------
    # set up labels
    if in_wavelengths:
        xlabel = 'wavelength (nm)'
    else:
        xlabel = 'order number + col (pixel)'
    ylabel = 'Degree of polarization for Stokes {0} (%)'.format(stokes)
    # set up title
    title = 'Polarimetry: Stokes {0}, Method={1}, for {2} exposures'
    titleargs = [stokes, method, nexp]
    # ---------------------------------------------------------------------
    # plot polarimetry data
    frame.plot(wl, pol, label='Degree of Polarization')
    # plot null1 data
    frame.plot(wl, null1, label='Null Polarization 1', linewidth=0.5, alpha=0.6)
    # plot null2 data
    frame.plot(wl, null2, label='Null Polarization 2', linewidth=0.5, alpha=0.6)
    # ---------------------------------------------------------------------
    # set title and labels
    frame.set(title=title.format(*titleargs), xlabel=xlabel, ylabel=ylabel)
    # ---------------------------------------------------------------------
    # plot legend
    frame.legend(loc=0)
    # ---------------------------------------------------------------------
    # end plotting function properly
    end_plotting(p, plot_name)


def polar_stokes_i_plot(p, loc, in_wavelengths=True):
    plot_name = 'polar_stokes_i_plot'
    # get data from loc
    wl, stokes_i = loc['FLAT_X'], loc['FLAT_STOKESI'] * loc['CONT_FLUX']
    contxbin, contybin = np.array(loc['CONT_FLUX_XBIN']), np.array(loc['CONT_FLUX_YBIN'])
    stokes_ierr = loc['FLAT_STOKESIERR'] * loc['CONT_FLUX']
    stokes = 'I'
    method, nexp = loc['METHOD'], loc['NEXPOSURES']
    # ---------------------------------------------------------------------
    # set up fig
    fig, frame = setup_figure(p)
    # ---------------------------------------------------------------------
    # set up labels
    if in_wavelengths:
        xlabel = 'wavelength (nm)'
    else:
        xlabel = 'order number + col (pixel)'
    ylabel = 'Stokes {0} total flux (ADU)'.format(stokes)
    # set up title
    title = 'Polarimetry: Stokes {0}, Method={1}, for {2} exposures'
    titleargs = [stokes, method, nexp]
    # ---------------------------------------------------------------------
    # plot stokes I data
    frame.errorbar(wl, stokes_i, yerr=stokes_ierr, linestyle='None', fmt='.', label='Stokes I')
    # plot continuum sample points
    frame.plot(contxbin, contybin, linestyle='None', marker='o', label='Continuum Samples')
    # plot continuum flux
    frame.plot(wl, loc['CONT_FLUX'], label='Continuum Flux for Normalization')
    # ---------------------------------------------------------------------

    # ---------------------------------------------------------------------
    # set title and labels
    frame.set(title=title.format(*titleargs), xlabel=xlabel, ylabel=ylabel)
    # ---------------------------------------------------------------------
    # plot legend
    frame.legend(loc=0)
    # end plotting function properly
    end_plotting(p, plot_name)


def clean_polarimetry_data(loc, sigclip=False, nsig=3, overwrite=False):
    """
    Function to clean polarimetry data.
    
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
            loc['WAVE']: numpy array (1D), wavelength data
            loc['STOKESI']: numpy array (1D), Stokes I data
            loc['STOKESIERR']: numpy array (1D), errors of Stokes I
            loc['POL']: numpy array (1D), degree of polarization data
            loc['POLERR']: numpy array (1D), errors of polarization
            loc['NULL2']: numpy array (1D), 2nd null polarization
        
    """

    # func_name = __NAME__ + '.clean_polarimetry_data()'

    loc['CLEAN_WAVE'], loc['CLEAN_STOKESI'], loc['CLEAN_STOKESIERR'] = [], [], []
    loc['CLEAN_POL'], loc['CLEAN_POLERR'], loc['CLEAN_NULL1'], loc['CLEAN_NULL2'] = [], [], [], []
    loc['CLEAN_CONT_POL'], loc['CLEAN_CONT_FLUX'] = [], []

    # get the shape of pol
    ydim, xdim = loc['POL'].shape

    # loop over each order
    for order_num in range(ydim):
        # mask NaN values
        mask = ~np.isnan(loc['POL'][order_num])
        mask &= ~np.isnan(loc['STOKESI'][order_num])
        mask &= ~np.isnan(loc['NULL1'][order_num])
        mask &= ~np.isnan(loc['NULL2'][order_num])
        mask &= ~np.isnan(loc['STOKESIERR'][order_num])
        mask &= ~np.isnan(loc['POLERR'][order_num])
        mask &= loc['STOKESI'][order_num] > 0
        mask &= ~np.isinf(loc['POL'][order_num])
        mask &= ~np.isinf(loc['STOKESI'][order_num])
        mask &= ~np.isinf(loc['NULL1'][order_num])
        mask &= ~np.isinf(loc['NULL2'][order_num])
        mask &= ~np.isinf(loc['STOKESIERR'][order_num])
        mask &= ~np.isinf(loc['POLERR'][order_num])
        
        if sigclip :
            median_pol = np.median(loc['POL'][order_num][mask])
            medsig_pol = np.median(np.abs(loc['POL'][order_num][mask] - median_pol))  / 0.67499
            mask &= loc['POL'][order_num] > median_pol - nsig * medsig_pol
            mask &= loc['POL'][order_num] < median_pol + nsig * medsig_pol

        wl = loc['WAVE'][order_num][mask]
        pol = loc['POL'][order_num][mask]
        polerr = loc['POLERR'][order_num][mask]
        flux = loc['STOKESI'][order_num][mask]
        fluxerr = loc['STOKESIERR'][order_num][mask]
        null1 = loc['NULL1'][order_num][mask]
        null2 = loc['NULL2'][order_num][mask]
        cont_pol = loc['ORDER_CONT_POL'][order_num][mask]
        cont_flux = loc['ORDER_CONT_FLUX'][order_num][mask]
        # test if order is not empty
        if len(wl):
            # append data to output vector
            loc['CLEAN_WAVE'] = np.append(loc['CLEAN_WAVE'], wl)
            loc['CLEAN_STOKESI'] = np.append(loc['CLEAN_STOKESI'], flux)
            loc['CLEAN_STOKESIERR'] = np.append(loc['CLEAN_STOKESIERR'], fluxerr)
            loc['CLEAN_POL'] = np.append(loc['CLEAN_POL'], pol)
            loc['CLEAN_POLERR'] = np.append(loc['CLEAN_POLERR'], polerr)
            loc['CLEAN_NULL1'] = np.append(loc['CLEAN_NULL1'], null1)
            loc['CLEAN_NULL2'] = np.append(loc['CLEAN_NULL2'], null2)
                
            loc['CLEAN_CONT_POL'] = np.append(loc['CLEAN_CONT_POL'], cont_pol)
            loc['CLEAN_CONT_FLUX'] = np.append(loc['CLEAN_CONT_FLUX'], cont_flux)

        if overwrite :
            loc['WAVE'][order_num][~mask] = np.nan
            loc['POL'][order_num][~mask] = np.nan
            loc['POLERR'][order_num][~mask] = np.nan
            loc['STOKESI'][order_num][~mask] = np.nan
            loc['STOKESIERR'][order_num][~mask] = np.nan
            loc['NULL1'][order_num][~mask] = np.nan
            loc['NULL2'][order_num][~mask] = np.nan


    # sort by wavelength (or pixel number)
    sortmask = np.argsort(loc['CLEAN_WAVE'])

    # save back to loc
    loc['FLAT_X'] = deepcopy(loc['CLEAN_WAVE'][sortmask])
    loc['FLAT_POL'] = deepcopy(loc['CLEAN_POL'][sortmask])
    loc['FLAT_POLERR'] = deepcopy(loc['CLEAN_POLERR'][sortmask])
    loc['FLAT_STOKESI'] = deepcopy(loc['CLEAN_STOKESI'][sortmask])
    loc['FLAT_STOKESIERR'] = deepcopy(loc['CLEAN_STOKESIERR'][sortmask])
    loc['FLAT_NULL1'] = deepcopy(loc['CLEAN_NULL1'][sortmask])
    loc['FLAT_NULL2'] = deepcopy(loc['CLEAN_NULL2'][sortmask])
    
    loc['CONT_POL'] = deepcopy(loc['CLEAN_CONT_POL'][sortmask])
    loc['CONT_FLUX'] = deepcopy(loc['CLEAN_CONT_FLUX'][sortmask])

    return loc


def save_pol_le_format(output, loc) :
    # The columns of LE .s format are the following
    # wavelength, I/Ic, V/Ic, Null1/Ic, Null2/Ic
    # e.g. 369.7156  2.8760e-01  3.0819e-02 -2.4229e-02  2.7975e-02  3.0383e-02

    loc = clean_polarimetry_data(loc)
    
    data_string = ""
    
    for i in range(len(loc['CLEAN_POL'])):
        wl = loc['CLEAN_WAVE'][i]
        pol, polerr = loc['CLEAN_POL'][i], loc['CLEAN_POLERR'][i]
        null1, null2 = loc['CLEAN_NULL1'][i], loc['CLEAN_NULL2'][i]
        stokesI = loc['CLEAN_STOKESI'][i]

        data_string += "{0:.4f} {1:.4e} {2:.4e} {3:.4e} {4:.4e} {5:.4e}\n".format(wl, stokesI, pol, polerr, null1, null2)

    out_string = "***Reduced spectrum of '{0}'\n".format(loc['OBJECT'])
    out_string += "{0} 5\n".format(len(loc['CLEAN_POL']))
    out_string += data_string
    
    outfile = open(output,"w+")
    outfile.write(out_string)
    outfile.close()


#--- Load a spirou spectrum from e.fits or t.fits file (which are the default products at CADC)
# This function preserves the spectral order structure
def load_spirou_AB_efits_spectrum(input, nan_pos_filter=True) :
    
    # open fits file
    hdu = fits.open(input)
    
    if input.endswith("e.fits") :
        WaveAB = hdu[5].data
        FluxAB = hdu[1].data
        #BlazeAB = hdu[9].data / np.median(hdu[9].data)
        BlazeAB = hdu[9].data / np.nanmean(hdu[9].data)

        WaveC = hdu[8].data
        FluxC = hdu[4].data
        #BlazeC = hdu[12].data / np.median(hdu[12].data)
        BlazeC = hdu[12].data / np.nanmean(hdu[12].data)

    elif input.endswith("t.fits") :
        WaveAB = hdu[2].data
        FluxAB = hdu[1].data
        #BlazeAB = hdu[3].data / np.median(hdu[3].data)
        BlazeAB = hdu[3].data / np.nanmean(hdu[3].data)
        Recon = hdu[4].data
    else :
        print("ERROR: input file type not recognized")
        exit()

    WaveABout, FluxABout, BlazeABout = [], [], []
    WaveCout, FluxCout, BlazeCout = [], [], []
    Reconout = []
    for i in range(len(WaveAB)) :
        if nan_pos_filter :
            # mask NaN values
            nanmask = np.where(~np.isnan(FluxAB[i]))
            # mask negative and zero values
            negmask = np.where(FluxAB[i][nanmask] > 0)

            WaveABout.append(WaveAB[i][nanmask][negmask])
            FluxABout.append(FluxAB[i][nanmask][negmask])
            BlazeABout.append(BlazeAB[i][nanmask][negmask])
            if input.endswith("e.fits") :
                WaveCout.append(WaveC[i][nanmask][negmask])
                FluxCout.append(FluxC[i][nanmask][negmask])
                BlazeCout.append(BlazeC[i][nanmask][negmask])
            elif input.endswith("t.fits") :
                Reconout.append(Recon[i][nanmask][negmask])
        else :
            WaveABout.append(WaveAB[i])
            FluxABout.append(FluxAB[i])
            BlazeABout.append(BlazeAB[i])
            if input.endswith("e.fits") :
                WaveCout.append(WaveC[i])
                FluxCout.append(FluxC[i])
                BlazeCout.append(BlazeC[i])
            elif input.endswith("t.fits") :
                Reconout.append(Recon[i])

    loc = {}
    loc['filename'] = input
    loc['header0'] = hdu[0].header
    loc['header1'] = hdu[1].header

    loc['WaveAB'] = WaveABout
    loc['FluxAB'] = FluxABout
    loc['BlazeAB'] = BlazeABout
    
    if input.endswith("e.fits") :
        loc['WaveC'] = WaveCout
        loc['FluxC'] = FluxCout
        loc['BlazeC'] = BlazeCout
    elif input.endswith("t.fits") :
        loc['Recon'] = Reconout

    return loc


def save_pol_fits(filename, p, loc) :
    
    header = loc['HEADER0']
    header1 = loc['HEADER1']
    
    header.set('ORIGIN', "spirou_pol")

    # get the shape of pol
    ydim, xdim = loc['POL'].shape
    maxlen = 0
    for order_num in range(ydim):
        if len(loc['POL'][order_num]) > maxlen :
            maxlen = len(loc['POL'][order_num])

    pol_data = np.full((ydim,maxlen), np.nan)
    polerr_data = np.full((ydim,maxlen), np.nan)

    stokesI_data = np.full((ydim,maxlen), np.nan)
    stokesIerr_data = np.full((ydim,maxlen), np.nan)

    null1_data = np.full((ydim,maxlen), np.nan)
    null2_data = np.full((ydim,maxlen), np.nan)

    wave_data = np.full((ydim,maxlen), np.nan)

    for order_num in range(ydim) :
        for i in range(len(loc['POL'][order_num])) :
            pol_data[order_num][i] = loc['POL'][order_num][i]
            polerr_data[order_num][i] = loc['POLERR'][order_num][i]

            stokesI_data[order_num][i] = loc['STOKESI'][order_num][i]
            stokesIerr_data[order_num][i] = loc['STOKESIERR'][order_num][i]

            null1_data[order_num][i] = loc['NULL1'][order_num][i]
            null2_data[order_num][i] = loc['NULL2'][order_num][i]

            wave_data[order_num][i] = loc['WAVE'][order_num][i]

    header.set('TTYPE1', "Pol")
    header.set('TUNIT1', "DEG")

    header.set('TTYPE2', "PolErr")
    header.set('TUNIT2', "DEG")

    header.set('TTYPE3', "StokesI")
    header.set('TUNIT3', "COUNTS")

    header.set('TTYPE4', "StokesIErr")
    header.set('TUNIT4', "COUNTS")

    header.set('TTYPE5', "Null1")
    header.set('TUNIT5', "DEG")

    header.set('TTYPE6', "Null2")
    header.set('TUNIT6', "DEG")

    header.set('TTYPE7', "WaveAB")
    header.set('TUNIT7', "NM")

    header = polar_header(p, loc, header)
    header1 = polar_header(p, loc, header1)

    loc['HEADER0'] = header
    loc['HEADER1'] = header1
    
    primary_hdu = fits.PrimaryHDU(header=header)

    hdu_pol = fits.ImageHDU(data=pol_data, name="Pol", header=header1)
    hdu_polerr = fits.ImageHDU(data=polerr_data, name="PolErr")

    hdu_stokesI = fits.ImageHDU(data=stokesI_data, name="StokesI", header=header1)
    hdu_stokesIerr = fits.ImageHDU(data=stokesIerr_data, name="StokesIErr")

    hdu_null1 = fits.ImageHDU(data=null1_data, name="Null1", header=header1)
    hdu_null2 = fits.ImageHDU(data=null2_data, name="Null2", header=header1)

    hdu_wave = fits.ImageHDU(data=wave_data, name="WaveAB")

    mef_hdu = fits.HDUList([primary_hdu, hdu_pol, hdu_polerr, hdu_stokesI, hdu_stokesIerr, hdu_null1, hdu_null2, hdu_wave])

    mef_hdu.writeto(filename, overwrite=True)


def polar_header(p, loc, hdr):
    """
        Function to construct header keywords to be saved in the polar products
        
        :param p: parameter dictionary, ParamDict containing constants
        
        :param loc: parameter dictionary, ParamDict containing data
        
        :param hdr: ParamDict, FITS header dictionary

        :return hdr: ParamDict, updated FITS header dictionary
    """
    
    hdr.set('ELAPSED_TIME', loc['ELAPSED_TIME'], 'Total elapsed time (s)')
    hdr.set('MJDCEN', loc['MJDCEN'], 'MJD at center of 4 exposures')
    hdr.set('BJDCEN', loc['BJDCEN'], 'BJD at center of 4 exposures')
    hdr.set('BERVCEN', loc['BERVCEN'], 'BERV at center of 4 exposures')
    hdr.set('BERVMAX', loc['BERVMAX'], 'Maximum BERV value')
    hdr.set('MEANBJD', loc['MEANBJD'], 'Mean BJD of 4 exposures')
    
    hdr.set('STOKES', loc['STOKES'], 'Stokes parameter')

    return hdr


def load_pol_fits(filename, loc) :

    hdu = fits.open(filename)
    
    header = hdu[0].header
    header1 = hdu[1].header
    
    loc['HEADER0'] = header
    loc['HEADER1'] = header1
    
    loc['POL'] = hdu['Pol'].data
    loc['POLERR'] = hdu['PolErr'].data
    loc['STOKESI'] = hdu['StokesI'].data
    loc['STOKESIERR'] = hdu['StokesIErr'].data
    loc['NULL1'] = hdu['Null1'].data
    loc['NULL2'] = hdu['Null2'].data
    loc['WAVE'] = hdu['WaveAB'].data

    if 'STOKES' in header.keys() :
        loc['STOKES'] = header['STOKES']
    else:
        loc['STOKES'] = ""
    return loc


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