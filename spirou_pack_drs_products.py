# -*- coding: iso-8859-1 -*-
"""
    Created on April 27 2020
    
    Description: This routine reads SPIRou DRS products and packs into CADC e.fits and t.fits formats
    
    @author: Eder Martioli <martioli@iap.fr>
    
    Institut d'Astrophysique de Paris, France.
    
    Simple usage example:
    
    # example using pattern to select files with relative path:
    python spirou_pack_drs_products.py --inputABpattern=*o_pp_e2dsff_AB.fits --blaze=2019-06-17_2426231f_pp_blaze -v
    
    ############
    For e.fits :
    ############

    # List of input DRS files:
    
    2426165o_pp_e2dsff_AB.fits
    2426165o_pp_e2dsff_A.fits
    2426165o_pp_e2dsff_B.fits
    2426165o_pp_e2dsff_C.fits
    
    2019-06-17_2426231f_pp_blaze_AB.fits
    2019-06-17_2426231f_pp_blaze_A.fits
    2019-06-17_2426231f_pp_blaze_B.fits
    2019-06-17_2426231f_pp_blaze_C.fits
    
    # Format for output e.fits CADC file:
    In [15]: hdu.info()
    Filename: 2426165e.fits
    No.    Name      Ver    Type      Cards   Dimensions   Format
    0  PRIMARY       1 PrimaryHDU     494   ()
    1  FluxAB        1 ImageHDU      1322   (4088, 49)   float64
    2  FluxA         1 ImageHDU      1322   (4088, 49)   float64
    3  FluxB         1 ImageHDU      1322   (4088, 49)   float64
    4  FluxC         1 ImageHDU       832   (4088, 49)   float64
    5  WaveAB        1 ImageHDU        16   (4088, 49)   float64
    6  WaveA         1 ImageHDU        16   (4088, 49)   float64
    7  WaveB         1 ImageHDU        16   (4088, 49)   float64
    8  WaveC         1 ImageHDU        16   (4088, 49)   float64
    9  BlazeAB       1 ImageHDU        17   (4088, 49)   float64
    10  BlazeA        1 ImageHDU        17   (4088, 49)   float64
    11  BlazeB        1 ImageHDU        17   (4088, 49)   float64
    12  BlazeC        1 ImageHDU        17   (4088, 49)   float64

    ############
    For t.fits :
    ############
    
    # List of input DRS files:

    2426165o_pp_e2dsff_tcorr_AB.fits
    2019-06-17_2426231f_pp_blaze_AB.fits
    2426165o_pp_e2dsff_recon_AB.fits

    # Format for output t.fits CADC file:
    In [17]: hdu.info()
    Filename: 2426165t.fits
    No.    Name      Ver    Type      Cards   Dimensions   Format
    0  PRIMARY       1 PrimaryHDU     494   ()
    1  FluxAB        1 ImageHDU      1332   (4088, 49)   float64
    2  WaveAB        1 ImageHDU        16   (4088, 49)   float64
    3  BlazeAB       1 ImageHDU        17   (4088, 49)   float64
    4  Recon         1 ImageHDU      1338   (4088, 49)   float64

    """

__version__ = "1.0"

__copyright__ = """
    Copyright (c) ...  All rights reserved.
    """

from optparse import OptionParser
import os,sys

import glob

import numpy as np
import astropy.io.fits as fits


def fits2wave(file_or_header):
    info = """
        Provide a fits header or a fits file
        and get the corresponding wavelength
        grid from the header.
        
        Usage :
        wave = fits2wave(hdr)
        or
        wave = fits2wave('my_e2ds.fits')
        
        Output has the same size as the input
        grid. This is derived from NAXIS
        values in the header
        """
    
    
    # check that we have either a fits file or an astropy header
    if type(file_or_header) == str:
        hdr = fits.getheader(file_or_header)
    elif str(type(file_or_header)) == "<class 'astropy.io.fits.header.Header'>":
        hdr = file_or_header
    else:
        print()
        print('~~~~ wrong type of input ~~~~')
        print()
        
        print(info)
        return []

    # get the keys with the wavelength polynomials
    wave_hdr = hdr['WAVE0*']
    # concatenate into a numpy array
    wave_poly = np.array([wave_hdr[i] for i in range(len(wave_hdr))])
    
    # get the number of orders
    nord = hdr['WAVEORDN']
    
    # get the per-order wavelength solution
    wave_poly = wave_poly.reshape(nord, len(wave_poly) // nord)
    
    # get the length of each order (normally that's 4088 pix)
    npix = hdr['NAXIS1']
    
    # project polynomial coefficiels
    wavesol = [np.polyval(wave_poly[i][::-1],np.arange(npix)) for i in range(nord) ]
    
    # return wave grid
    return np.array(wavesol)


def set_filenames(ref_AB_flux_filename, blaze) :
    efiles, tfiles = {}, {}
    
    """
        Description: function to figure out the name of files associated to
        an input reference file
    e.g.:
        efiles, tfiles = set_filenames("2426165o_pp_e2dsff_AB.fits","2019-06-17_2426231f_pp_blaze")
       
    efiles:
    
    2426165o_pp_e2dsff_AB.fits
    2426165o_pp_e2dsff_A.fits
    2426165o_pp_e2dsff_B.fits
    2426165o_pp_e2dsff_C.fits
    
    2019-06-17_2426231f_pp_blaze_AB.fits
    2019-06-17_2426231f_pp_blaze_A.fits
    2019-06-17_2426231f_pp_blaze_B.fits
    2019-06-17_2426231f_pp_blaze_C.fits
    
    ---
    
    tfiles:
    
    2426165o_pp_e2dsff_tcorr_AB.fits
    2019-06-17_2426231f_pp_blaze_AB.fits
    2426165o_pp_e2dsff_recon_AB.fits
    """
    
    core_name = ref_AB_flux_filename.replace("_AB.fits","")
    
    efiles['FluxAB'] = "{0}_AB.fits".format(core_name)
    efiles['FluxA'] = "{0}_A.fits".format(core_name)
    efiles['FluxB'] = "{0}_B.fits".format(core_name)
    efiles['FluxC'] = "{0}_C.fits".format(core_name)
    efiles['BlazeAB'] = "{0}_AB.fits".format(blaze)
    efiles['BlazeA'] = "{0}_A.fits".format(blaze)
    efiles['BlazeB'] = "{0}_B.fits".format(blaze)
    efiles['BlazeC'] = "{0}_C.fits".format(blaze)
    efiles['output'] = ref_AB_flux_filename.replace("o_pp_e2dsff_AB.fits","e.fits")

    tfiles['FluxAB'] = "{0}_tcorr_AB.fits".format(core_name)
    tfiles['BlazeAB'] = "{0}_AB.fits".format(blaze)
    tfiles['Recon'] = "{0}_recon_AB.fits".format(core_name)
    tfiles['output'] = ref_AB_flux_filename.replace("o_pp_e2dsff_AB.fits","t.fits")

    return efiles, tfiles


def make_th_lc_keywords(header) :
    """
        Description: function to create TH_* keywords for wavelength calibration
        as produced by previous versions of the DRS for backwards compatibility.
    """
    norders = header["WAVEORDN"]
    degpoly = header["WAVEDEGN"]
            
    header.set('TH_ORD_N', norders, "nb orders in total")
    header.set('TH_LL_D', degpoly, "deg polyn fit ll(x,order)")

    ncount = 0
    for order in range(norders) :
        for coeff in range(degpoly+1) :
            th_lc_key = "TH_LC{0}".format(ncount)
            wave_key = "WAVE{0:04d}".format(ncount)
            th_lc_comment = "coeff ll(x,order) order={0} coeff={1}".format(order, coeff)
            header.set(th_lc_key, header[wave_key], th_lc_comment)
            ncount += 1
    return header


def save_e_fits(efiles) :
    """
        Description: function to save e.fits file
        """
    outhdulist = []
    
    if os.path.exists(efiles['FluxAB']) :
        
        FluxAB_hdr = fits.getheader(efiles['FluxAB'], 0)
        
        primary_hdu = fits.PrimaryHDU(header=FluxAB_hdr)
        outhdulist.append(primary_hdu)
    else :
        print("ERROR: missing file:",efiles['FluxAB'])
        exit()

    # create flux extensions
    fluxkeys = ['FluxAB', 'FluxA', 'FluxB', 'FluxC']

    for key in fluxkeys :
        if os.path.exists(efiles[key]) :
            data, hdr = fits.getdata(efiles[key], 0, header=True)
            hdr = make_th_lc_keywords(hdr)
            outhdulist.append(fits.ImageHDU(data=data, name=key, header=hdr))
        else :
            print("ERROR: missing file:",efiles[key])
            exit()

    # Create Wavelength extensions
    wavekeys = ['WaveAB', 'WaveA', 'WaveB', 'WaveC']

    for key in wavekeys :
        
        flux_key = key.replace("Wave","Flux")
        
        if os.path.exists(efiles[flux_key]) :
            data = fits2wave(efiles[flux_key])
            outhdulist.append(fits.ImageHDU(data=data, name=key))
        else :
            print("ERROR: missing file:",efiles[flux_key])
            exit()

    # Create blaze extensions
    blazekeys = ['BlazeAB', 'BlazeA', 'BlazeB', 'BlazeC']

    for key in blazekeys :
        if os.path.exists(efiles[key]) :
            data, hdr = fits.getdata(efiles[key], 0, header=True)
            outhdulist.append(fits.ImageHDU(data=data, name=key))
        else :
            print("ERROR: missing file:",efiles[key])
            exit()

    mef_hdu = fits.HDUList(outhdulist)
    mef_hdu.writeto(efiles['output'], overwrite=True)

    return efiles['output']



def save_t_fits(tfiles) :
    """
        Description: function to save t.fits file
        """
    
    outhdulist = []
    
    if os.path.exists(tfiles['FluxAB']) :
        
        FluxAB_data, FluxAB_hdr = fits.getdata(tfiles['FluxAB'], 0, header=True)
        
        FluxAB_hdr = make_th_lc_keywords(FluxAB_hdr)
        
        primary_hdu = fits.PrimaryHDU(header=FluxAB_hdr)
        outhdulist.append(primary_hdu)
        
        outhdulist.append(fits.ImageHDU(data=FluxAB_data, name='FluxAB', header=FluxAB_hdr))
    
        WaveAB_data = fits2wave(tfiles['FluxAB'])
    
        outhdulist.append(fits.ImageHDU(data=WaveAB_data, name='WaveAB'))
    
    else :
        print("ERROR: missing file:",tfiles['FluxAB'])
        exit()


    if os.path.exists(tfiles['BlazeAB']) :
        data, hdr = fits.getdata(tfiles['BlazeAB'], 0, header=True)
        outhdulist.append(fits.ImageHDU(data=data, name='BlazeAB'))
    else :
        print("ERROR: missing file:",tfiles['BlazeAB'])
        exit()


    if os.path.exists(tfiles['Recon']) :
        data, hdr = fits.getdata(tfiles['Recon'], 0, header=True)
        outhdulist.append(fits.ImageHDU(data=data, name='Recon'))
    else :
        print("ERROR: missing file:",tfiles['Recon'])
        exit()

    mef_hdu = fits.HDUList(outhdulist)
    mef_hdu.writeto(tfiles['output'], overwrite=True)

    return tfiles['output']


def check_files_exist(files, verbose=False) :
    """
        Description: function to check if either t- or e-files exist
        return: True if all files exist or False otherwise
        """
    for key in files.keys() :
        if os.path.exists(files[key]) or key=='output':
            continue
        else :
            if verbose :
                print("File",files[key]," does not exists, skipping sequence ...")
            return False
    return True


parser = OptionParser()
parser.add_option("-i", "--inputABpattern", dest="inputABpattern", help="Spectral e2ds AB flux data pattern",type='string',default="*o_pp_e2dsff_AB.fits")
parser.add_option("-b", "--blaze", dest="blaze", help="e2ds blaze data root name",type='string',default="")
parser.add_option("-o", action="store_true", dest="overwrite", help="overwrite output fits", default=False)
parser.add_option("-v", action="store_true", dest="verbose", help="verbose", default=False)

try:
    options,args = parser.parse_args(sys.argv[1:])
except:
    print("Error: check usage with spirou_pack_drs_products.py -h ")
    sys.exit(1)

if options.verbose:
    print('Spectral e2ds AB flux data pattern: ', options.inputABpattern)
    print('Spectral e2ds blaze data root name: ', options.blaze)

# make list of efits data files
if options.verbose:
    print("Creating list of e2ds AB spectrum files...")
inputABdata = sorted(glob.glob(options.inputABpattern))


for i in range(len(inputABdata)) :
    
    efiles, tfiles = set_filenames(inputABdata[i],options.blaze)
    
    if check_files_exist(efiles, options.verbose) :
        if options.verbose :
            print("Creating file {0}/{1}: {2} ".format(i,len(inputABdata),efiles["output"]))
        save_e_fits(efiles)

    if check_files_exist(tfiles, options.verbose) :
        if options.verbose :
            print("Creating file {0}/{1}: {2} ".format(i,len(inputABdata),tfiles["output"]))
        save_t_fits(tfiles)
