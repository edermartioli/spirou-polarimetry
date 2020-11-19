# -*- coding: utf-8 -*-
"""
    Spirou packager for polarimetry module
    
    Created on 2020-10-21
    
    @author: E. Martioli, C. Usher
    """

import textwrap
from typing import Collection, Iterable, Union
from astropy.io import fits
import numpy as np
from spirouPolar import polar_header

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

    for order_num in range(ydim) :
        for i in range(len(loc['POL'][order_num])) :
            pol_data[order_num][i] = loc['POL'][order_num][i]
            polerr_data[order_num][i] = loc['POLERR'][order_num][i]

            stokesI_data[order_num][i] = loc['STOKESI'][order_num][i]
            stokesIerr_data[order_num][i] = loc['STOKESIERR'][order_num][i]

            null1_data[order_num][i] = loc['NULL1'][order_num][i]
            null2_data[order_num][i] = loc['NULL2'][order_num][i]

            wave_data[order_num][i] = loc['WAVE'][order_num][i]

    loc['pol_data'] = pol_data
    loc['polerr_data'] = polerr_data

    loc['stokesI_data'] = stokesI_data
    loc['stokesIerr_data'] = stokesIerr_data

    loc['null1_data'] = null1_data
    loc['null2_data'] = null2_data

    loc['wave_data'] = wave_data

    return loc


def add_polar_keywords(p, loc, hdr):
    return polar_header(p, loc, hdr)


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
