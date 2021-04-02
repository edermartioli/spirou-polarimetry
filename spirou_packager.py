# -*- coding: utf-8 -*-
"""
    Spirou packager for polarimetry module
    
    Created on 2020-10-21
    
    @author: E. Martioli, C. Usher
    """

import  os
import textwrap
from typing import Collection, Iterable, Union
from astropy.io import fits
import numpy as np

ExtensionHDU = Union[fits.ImageHDU, fits.BinTableHDU]
HDU = Union[fits.PrimaryHDU, ExtensionHDU]

def create_hdu_list(hdus: Collection[HDU]) -> fits.HDUList:
    """
    Takes a collection of fits HDUs and converts into an HDUList ready to be saved to a file
    :param hdus: The collection of fits HDUs
    :return: A fits HDUList ready to be saved to a file
    """
    hdu_list = fits.HDUList()
    for hdu in hdus:
        if len(hdu_list) == 0:
            hdu.header['NEXTEND'] = len(hdus) - 1
        else:
            hdu.header.remove('NEXTEND', ignore_missing=True)
        hdu_list.append(hdu)
    return hdu_list


def remove_keys(header: fits.Header, keys: Iterable[str]):
    """
    Removes any of the specified keys from a fits header, if present.
    :param header: The fits header to update
    :param keys: The keys to remove
    """
    for key in keys:
        header.remove(key, ignore_missing=True)


def product_header_update(hdu_list: fits.HDUList):
    """
    Puts the finishing touches on the product header, which currently consists of:
    1. Removing data dimension keys from primary header
    2. Copying VERSION key from first extension to primary header
    3. For p.fits copying EXPTIME/MJDATE from Pol extension to primary header
    4. For non-calibration non-Err extensions, removing cards which duplicates from the primary header.
    5. Adding a COMMENT to the primary header listing the EXTNAME of each extension.
    :param hdu_list: HDUList to update
    """
    if len(hdu_list) <= 1:
        print('ERROR: Trying to create product primary HDU with no extensions')
        return
    primary_header = hdu_list[0].header
    remove_keys(primary_header, ('BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2'))
    primary_header.remove('COMMENT', ignore_missing=True, remove_all=True)

    ext_names = []
    for extension in hdu_list[1:]:
        ext_header = extension.header
        ext_name = ext_header['EXTNAME']
        ext_names.append(ext_name)
    description = 'This file contains the following extensions: ' + ', '.join(ext_names)
    for line in textwrap.wrap(description, 71):
        primary_header.insert('FILENAME', ('COMMENT', line))


def extension_from_hdu(ext_name: str, hdu: HDU) -> ExtensionHDU:
    """
        Takes a fits HDU and inserts the EXTNAME at the at the first available spot in the header.
        :param ext_name: The value for EXTNAME
        :param hdu: The HDU to update
        :return: The updated HDU
        """
    if ext_name:
        hdu.header.remove('EXTNAME', ignore_missing=True)
        extname_card = ('EXTNAME', ext_name)
        if 'XTENSION' in hdu.header:
            if 'TFIELDS' in hdu.header:
                hdu.header.insert('TFIELDS', extname_card, after=True)
            else:
                hdu.header.insert('GCOUNT', extname_card, after=True)
        else:
            hdu.header.insert(0, extname_card)
    return hdu


def make_2D_data_uniform(loc):
    """
        Takes all polarimetry data arrays in loc and make them uniform,
        i.e. with the same shape by filling voids with np.nans
        :param loc: dict with data arrays and other polarimetry info
        :return: The updated loc
        """

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
    blaze_data = np.full((ydim,maxlen), np.nan)

    for order_num in range(ydim) :
        for i in range(len(loc['POL'][order_num])) :
            pol_data[order_num][i] = loc['POL'][order_num][i]
            polerr_data[order_num][i] = loc['POLERR'][order_num][i]

            stokesI_data[order_num][i] = loc['STOKESI'][order_num][i]
            stokesIerr_data[order_num][i] = loc['STOKESIERR'][order_num][i]

            null1_data[order_num][i] = loc['NULL1'][order_num][i]
            null2_data[order_num][i] = loc['NULL2'][order_num][i]

            wave_data[order_num][i] = loc['WAVE'][order_num][i]
            if 'BLAZE' in loc.keys() :
                blaze_data[order_num][i] = loc['BLAZE'][order_num][i]

    loc['pol_data'] = pol_data
    loc['polerr_data'] = polerr_data

    loc['stokesI_data'] = stokesI_data
    loc['stokesIerr_data'] = stokesIerr_data

    loc['null1_data'] = null1_data
    loc['null2_data'] = null2_data

    loc['wave_data'] = wave_data
    loc['blaze_data'] = blaze_data

    return loc


def add_polar_keywords(p, loc, hdr, apero=False):
    return polar_header(p, loc, hdr, apero=apero)


def create_pol_product(product, p, loc):
    """
    Create the p.fits product:
    Polarimetric products only processed in polarimetric mode, from the combination of 4 consecutive exposures.
    HDU #    Name        Type             Description
        1                Primary Header
        2   Pol          Image            The polarized spectrum in the required Stokes configuration
        3   PolErr       Image            The error on the polarized spectrum
        4   StokesI      Image            The combined Stokes I (intensity) spectrum
        5   StokesIErr   Image            The error on the Stokes I spectrum
        6   Null1        Image            One null spectrum used to check the polarized signal (see Donati et al 1997)
        7   Null2        Image            The other null spectrum used to check the polarized signal
        8   WaveAB       Image            The wavelength vector for the AB science channel
        9   BlazeAB      Image            The Blaze function for AB (useful for Stokes I)
    :param exposure: Exposure to create product for
    """

    efits = loc['BASENAME']
    
    def wipe_snr(header):
        for key in header:
            if key.startswith('EXTSN'):
                header[key] = 'Unknown'

    print('INFO: Creating {}'.format(product))
    try:
        # update loc to make all data 2D arrays uniform
        loc = make_2D_data_uniform(loc)
        
        # open e.fits file of first image in the sequence
        hdu_list = fits.open(efits)
        
        # same primary hdu as in first efits of sequence:
        primary_hdu = hdu_list[0]
    
        hdu_wave = fits.ImageHDU(data=loc['wave_data'], header=hdu_list['WaveAB'].header, name='WaveAB')

        # same cal extensions as in first efits of sequence:
        cal_extensions = [
            extension_from_hdu('WaveAB', hdu_wave),
            extension_from_hdu('BlazeAB', hdu_list['BlazeAB'])
        ]
        
        hdu_pol = fits.ImageHDU(data=loc['pol_data'], header=hdu_list['FluxA'].header, name='Pol')
        hdu_pol.header = add_polar_keywords(p, loc, hdu_pol.header)
        hdu_polerr = fits.ImageHDU(data=loc['polerr_data'], name='PolErr')

        hdu_stokesI = fits.ImageHDU(data=loc['stokesI_data'], header=hdu_list['FluxA'].header, name='StokesI')
        hdu_stokesI.header = add_polar_keywords(p, loc, hdu_stokesI.header)
        hdu_stokesIerr = fits.ImageHDU(data=loc['stokesIerr_data'], name='StokesIErr')
        
        hdu_null1 = fits.ImageHDU(data=loc['null1_data'], header=hdu_list['FluxA'].header, name='Null1')
        hdu_null1.header = add_polar_keywords(p, loc, hdu_null1.header)
        hdu_null2 = fits.ImageHDU(data=loc['null2_data'], header=hdu_list['FluxA'].header, name='Null2')
        hdu_null2.header = add_polar_keywords(p, loc, hdu_null2.header)

        pol_extensions = [
            extension_from_hdu('Pol', hdu_pol),
            extension_from_hdu('PolErr', hdu_polerr),
            extension_from_hdu('StokesI', hdu_stokesI),
            extension_from_hdu('StokesIErr', hdu_stokesIerr),
            extension_from_hdu('Null1', hdu_null1),
            extension_from_hdu('Null2', hdu_null2),
        ]
        for ext in pol_extensions:
            wipe_snr(ext.header)

        hdu_list = create_hdu_list([primary_hdu, *pol_extensions, *cal_extensions])
        product_header_update(hdu_list)

        # We copy input files to primary header after duplicate keys have been cleaned out
        primary_header = hdu_list[0].header
        pol_header = hdu_list[1].header
        in_file_cards = [card for card in pol_header.cards if card[0].startswith('FILENAM')]
        for card in in_file_cards:
            primary_header.insert('FILENAME', card)
        primary_header.remove('FILENAME', ignore_missing=True)

        hdu_list.writeto(product, overwrite=True)
    except:
        print('ERROR: Creation of {} failed'.format(product))


def polar_header(p, loc, hdr, apero=False):
    """
        Function to add polarimetry keywords in the header of polar products
        
        :param p: parameter dictionary, ParamDict containing constants
        
        :param loc: parameter dictionary, ParamDict containing data
        
        :param hdr: ParamDict, FITS header dictionary

        :return hdr: ParamDict, updated FITS header dictionary
    """
    
    polardict = loc['POLARDICT']

    
    ########################
    # keywords set as placeholder, but without meaning since it's run outside the DRS
    ########################
    hdr.set('DRS_EOUT', 'OBJ_FP  ', 'DRS Extraction input DPRTYPE')
    #hdr.set('DRSPID', 'None', 'The process ID that outputted this file.')
    hdr.set('WAVELOC', 'WaveAB', 'Where the wave solution was read from')
    hdr.set('QCC', 1, 'All quality control passed')
    hdr.set('QCC001N', 'None    ', 'Quality control variable name')
    hdr.set('QCC001V', 'None    ', 'Qualtity control value')
    hdr.set('QCC001L', 'None    ', 'Quality control logic')
    hdr.set('QCC001P', 1, 'Quality control passed')
    ########################
    if apero :
        for exp in polardict["INPUT_EXPOSURES"]:
            # get expnum
            expnum = polardict[exp]['exposure']
            # add keywords to inform which files have been used to create output
            # The header of e.fits already has INF1*, so I have used INF2*, but not sure it's correct
            infkey = "INF2{0:03d}".format(expnum)
            hdr.set(infkey, os.path.basename(exp), 'Input file used to create output file={}'.format(expnum))
    else :
        # loop over files in polar sequence to set *e.fits filenames into INF* keywords
        for filename in polardict.keys():
            # get expnum
            expnum = polardict[filename]['exposure']
        
            # add keywords to inform which files have been used to create output
            # The header of e.fits already has INF1*, so I have used INF2*, but not sure it's correct
            infkey = "INF2{0:03d}".format(expnum)
            hdr.set(infkey, os.path.basename(filename), 'Input file used to create output file={}'.format(expnum))

    ########################
    # add polarimetry related keywords, as in previous version:
    ########################
    hdr.set('ELAPTIME', loc['ELAPSED_TIME'], 'Elapsed time of observation (sec)')
    hdr.set('MJDCEN', loc['MJDCEN'], 'MJD at center of observation')
    hdr.set('BJDCEN', loc['BJDCEN'], 'BJD at center of observation')
    hdr.set('BERVCEN', loc['BERVCEN'], 'BERV at center of observation')
    hdr.set('MEANBJD', loc['MEANBJD'], 'Mean BJD for polar sequence')
    hdr.set('STOKES', loc['STOKES'], 'Stokes paremeter: Q, U, V, or I')
    hdr.set('POLNEXP', loc['NEXPOSURES'], 'Number of exposures for polarimetry')
    hdr.set('TOTETIME', loc['TOTEXPTIME'], 'Total exposure time (sec)')
    hdr.set('POL_DEG', 'POL_DEG ', 'DRS output identification code')
    hdr.set('POLMETHO', p['IC_POLAR_METHOD'], 'Polarimetry method')
    ########################

    ########################
    # suggested new keywords, which have only been introduced in the new version of polarimetry
    ########################
    hdr.set('MJDFWCEN', loc['MJDFWCEN'], 'MJD at flux-weighted center of 4 exposures')
    hdr.set('BJDFWCEN', loc['BJDFWCEN'], 'BJD at flux-weighted center of 4 exposures')
    hdr.set('MEANBERV', loc['MEANBERV'], 'Mean BERV of 4 exposures')
    hdr.set('TCORRFLX', p['IC_POLAR_USE_TELLURIC_CORRECTED_FLUX'], 'Polarimetry used tellcorr flux')
    hdr.set('CORRBERV', p['IC_POLAR_BERV_CORRECT'], 'BERV corrected before polarimetry')
    hdr.set('CORRSRV', p['IC_POLAR_SOURCERV_CORRECT'], 'Source RV corrected before polarimetry')
    hdr.set('NSTOKESI', p['IC_POLAR_NORMALIZE_STOKES_I'], 'Normalize Stokes I by continuum')
    hdr.set('PINTERPF', p['IC_POLAR_INTERPOLATE_FLUX'], 'Interp flux to correct for shifts between exps')
    hdr.set('PSIGCLIP', p['IC_POLAR_CLEAN_BY_SIGMA_CLIPPING'], 'Apply polarimetric sigma-clip cleaning')
    hdr.set('PNSIGMA', p['IC_POLAR_NSIGMA_CLIPPING'], 'Number of sigmas of sigma-clip cleaning')
    hdr.set('PREMCONT', p['IC_POLAR_REMOVE_CONTINUUM'], 'Remove continuum polarization')
    hdr.set('PCONTAL', p['IC_POLAR_CONTINUUM_DETECTION_ALGORITHM'], 'Polarization continuum detection algorithm')
    hdr.set('SICONTAL', p['IC_STOKESI_CONTINUUM_DETECTION_ALGORITHM'], 'Stokes I continuum detection algorithm')
    hdr.set('PCPOLFIT', p['IC_POLAR_CONT_POLYNOMIAL_FIT'], 'Use polynomial fit for continuum polarization')
    hdr.set('PCPOLDEG', p['IC_POLAR_CONT_DEG_POLYNOMIAL'], 'Degree of polynomial to fit continuum polariz.')
    hdr.set('SICFUNC', p['IC_STOKESI_IRAF_CONT_FIT_FUNCTION'], 'Function to fit Stokes I continuum')
    hdr.set('SIPOLDEG', p['IC_STOKESI_IRAF_CONT_FUNCTION_ORDER'], 'Degree of polynomial to fit Stokes I continuum')
    hdr.set('PCBINSIZ', p['IC_POLAR_CONT_BINSIZE'], 'Polarimetry continuum bin size')
    hdr.set('PCOVERLA', p['IC_POLAR_CONT_OVERLAP'], 'Polarimetry continuum overlap size')
    for i in range(len(p['IC_POLAR_CONT_TELLMASK'])):
        hdr.set('PCEWL{0:03d}'.format(i),"{0},{1}".format(p['IC_POLAR_CONT_TELLMASK'][i][0],p['IC_POLAR_CONT_TELLMASK'][i][1]), 'Excluded wave range (nm) for cont detection {0}/{1}'.format(i,len(p['IC_POLAR_CONT_TELLMASK'])-1))
    ###############

    if apero :
        for exp in polardict["INPUT_EXPOSURES"]:
            # get expnum
            expnum = polardict[exp]['exposure']
            
            entry = polardict[exp]
            # get header
            e2ds = entry["e2dsff_AB"]
            hdr = fits.getheader(e2ds)
            
            hdr.set("FILENAM{0:1d}".format(expnum), hdr['FILENAME'], 'Base filename of exposure {}'.format(expnum))
            hdr.set("EXPTIME{0:1d}".format(expnum), hdr['EXPTIME'], 'EXPTIME of exposure {} (sec)'.format(expnum))
            hdr.set("MJDATE{0:1d}".format(expnum), hdr['MJDATE'], 'MJD at start of exposure {}'.format(expnum))
            hdr.set("MJDEND{0:1d}".format(expnum), hdr['MJDEND'], 'MJDEND at end of exposure {}'.format(expnum))
            hdr.set("BJD{0:1d}".format(expnum), hdr['BJD'], 'BJD at start of exposure {}'.format(expnum))
            hdr.set("BERV{0:1d}".format(expnum), hdr['BERV'], 'BERV at start of exposure {}'.format(expnum))
    else :
        # loop over files in polar sequence to add keywords related to each exposure in sequence
        for filename in polardict.keys():
            # get expnum
            expnum = polardict[filename]['exposure']
            # get header
            ehdr0 = fits.getheader(filename,0)
            ehdr1 = fits.getheader(filename,1)

            hdr.set("FILENAM{0:1d}".format(expnum), ehdr0['FILENAME'], 'Base filename of exposure {}'.format(expnum))
            hdr.set("EXPTIME{0:1d}".format(expnum), ehdr0['EXPTIME'], 'EXPTIME of exposure {} (sec)'.format(expnum))
            hdr.set("MJDATE{0:1d}".format(expnum), ehdr0['MJDATE'], 'MJD at start of exposure {}'.format(expnum))
            hdr.set("MJDEND{0:1d}".format(expnum), ehdr0['MJDEND'], 'MJDEND at end of exposure {}'.format(expnum))
            hdr.set("BJD{0:1d}".format(expnum), ehdr1['BJD'], 'BJD at start of exposure {}'.format(expnum))
            hdr.set("BERV{0:1d}".format(expnum), ehdr1['BERV'], 'BERV at start of exposure {}'.format(expnum))

    return hdr


def apero_create_pol_product(product, p, loc):
    """
    Create the p.fits product:
    Polarimetric products only processed in polarimetric mode, from the combination of 4 consecutive exposures.
    HDU #    Name        Type             Description
        1                Primary Header
        2   Pol          Image            The polarized spectrum in the required Stokes configuration
        3   PolErr       Image            The error on the polarized spectrum
        4   StokesI      Image            The combined Stokes I (intensity) spectrum
        5   StokesIErr   Image            The error on the Stokes I spectrum
        6   Null1        Image            One null spectrum used to check the polarized signal (see Donati et al 1997)
        7   Null2        Image            The other null spectrum used to check the polarized signal
        8   WaveAB       Image            The wavelength vector for the AB science channel
        9   BlazeAB      Image            The Blaze function for AB (useful for Stokes I)
    :param exposure: Exposure to create product for
    """
    from apero import core
    # Get Logging function
    WLOG = core.wlog
    
    # load polardict from loc
    polardict = loc['POLARDICT']
    # get base entry
    entry_base = polardict[loc['BASENAME']]

    def wipe_snr(header):
        for key in header:
            if key.startswith('EXTSN'):
                header[key] = 'Unknown'

    WLOG(p, 'info', 'Creating {}'.format(product))
    
    try:
        # update loc to make all data 2D arrays uniform
        loc = make_2D_data_uniform(loc)
        
        # set key for e2ds fits files
        base_e2ds = entry_base["e2dsff_AB"]
        base_wave_e2ds = entry_base["WAVE_AB"]
        base_blaze_e2ds = entry_base["BLAZE_AB"]

        # open e2ds fits file of first image in the sequence
        hdu_list_base = fits.open(base_e2ds)
        hdr_base = hdu_list_base[0].header
        # open e2ds fits file of first image in the sequence
        hdu_list_wave = fits.open(base_wave_e2ds)
        hdr_wave_base = hdu_list_wave[0].header
        # open e2ds fits file of first image in the sequence
        hdu_list_blaze = fits.open(base_blaze_e2ds)
        hdr_blaze_base = hdu_list_blaze[0].header

        # same primary hdu as in first efits of sequence:
        primary_hdu = hdu_list_base[0]
    
        hdu_wave = fits.ImageHDU(data=loc['wave_data'], header=hdr_wave_base, name='WaveAB')
        hdu_wave.header = add_polar_keywords(p, loc, hdu_wave.header, apero=True)

        hdu_blaze = fits.ImageHDU(data=loc['blaze_data'], header=hdr_blaze_base, name='BlazeAB')
        hdu_blaze.header = add_polar_keywords(p, loc, hdu_blaze.header, apero=True)

        # same cal extensions as in first efits of sequence:
        cal_extensions = [
            extension_from_hdu('WaveAB', hdu_wave),
            extension_from_hdu('BlazeAB', hdu_blaze)
        ]
        
        hdu_pol = fits.ImageHDU(data=loc['pol_data'], header=hdr_base, name='Pol')
        hdu_pol.header = add_polar_keywords(p, loc, hdu_pol.header, apero=True)
        hdu_polerr = fits.ImageHDU(data=loc['polerr_data'], name='PolErr')

        hdu_stokesI = fits.ImageHDU(data=loc['stokesI_data'], header=hdr_base, name='StokesI')
        hdu_stokesI.header = add_polar_keywords(p, loc, hdu_stokesI.header, apero=True)
        hdu_stokesIerr = fits.ImageHDU(data=loc['stokesIerr_data'], name='StokesIErr')
        
        hdu_null1 = fits.ImageHDU(data=loc['null1_data'], header=hdr_base, name='Null1')
        hdu_null1.header = add_polar_keywords(p, loc, hdu_null1.header, apero=True)
        hdu_null2 = fits.ImageHDU(data=loc['null2_data'], header=hdr_base, name='Null2')
        hdu_null2.header = add_polar_keywords(p, loc, hdu_null2.header, apero=True)

        pol_extensions = [
            extension_from_hdu('Pol', hdu_pol),
            extension_from_hdu('PolErr', hdu_polerr),
            extension_from_hdu('StokesI', hdu_stokesI),
            extension_from_hdu('StokesIErr', hdu_stokesIerr),
            extension_from_hdu('Null1', hdu_null1),
            extension_from_hdu('Null2', hdu_null2),
        ]
        for ext in pol_extensions:
            wipe_snr(ext.header)

        hdu_list = create_hdu_list([primary_hdu, *pol_extensions, *cal_extensions])
        product_header_update(hdu_list)

        # We copy input files to primary header after duplicate keys have been cleaned out
        primary_header = hdu_list[0].header
        pol_header = hdu_list[1].header
        in_file_cards = [card for card in pol_header.cards if card[0].startswith('FILENAM')]
        for card in in_file_cards:
            primary_header.insert('FILENAME', card)
        primary_header.remove('FILENAME', ignore_missing=True)

        hdu_list.writeto(product, overwrite=True, output_verify="fix+warn")
    except:
        WLOG(p,'error','Creation of {} failed'.format(product))


