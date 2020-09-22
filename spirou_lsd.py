# -*- coding: iso-8859-1 -*-
"""
    Created on December 9 2019
    
    Description: This routine performs LSD analysis of SPIRou spectro-polarimetric data.
    
    @author: Eder Martioli <martioli@iap.fr>
    
    Institut d'Astrophysique de Paris, France.
    
    Simple usage example:
    
    GamEqu:
    
    python $PATH/spirou_lsd.py --input=2329699_pol.fits --lsdmask=$PATH/lsd_masks/marcs_t5000g50_all --output=2329699_lsd.fits -p

    """

__version__ = "1.0"

__copyright__ = """
    Copyright (c) ...  All rights reserved.
    """

from optparse import OptionParser
import os,sys

import spirouPolar, polar_param
import spirouLSD

parser = OptionParser()
parser.add_option("-i", "--input", dest="input", help="Input SPIRou spectro-polarimetry file",type='string',default="")
parser.add_option("-m", "--lsdmask", dest="lsdmask", help="LSD mask",type='string', default="")
parser.add_option("-o", "--output", dest="output", help="Output LSD file",type='string', default="")
parser.add_option("-p", action="store_true", dest="plot", help="plot", default=False)
parser.add_option("-v", action="store_true", dest="verbose", help="verbose", default=False)

try:
    options,args = parser.parse_args(sys.argv[1:])
except:
    print("Error: check usage with  -h spirou_lsd.py")
    sys.exit(1)

if options.verbose:
    print('Input SPIRou spectro-polarimetry file: ', options.input)
    print('LSD mask: ', options.lsdmask)
    print('Output file: ', options.output)

# set up data storage
loc = {}

# define polarimetry parameters
p = polar_param.load_polar_parameters()

# set input polarimetry file
loc['POL_FITS_FILE'] = options.input

# load files
loc = spirouPolar.load_pol_fits(options.input, loc)

# select LSD mask file
if options.lsdmask != "" :
    # set lsd mask file from input
    loc['LSD_MASK_FILE'] = options.lsdmask
else :
    # select an lsd mask file from repositories
    loc['LSD_MASK_FILE'] = spirouLSD.select_lsd_mask(p)

# ------------------------------------------------------------------
# Run LSD Analysis
# ------------------------------------------------------------------
loc = spirouLSD.lsd_analysis_wrapper(p, loc)

# save LSD data to fits
if options.output != "" :
    spirouLSD.save_lsd_fits(options.output, loc, p)

if options.plot :
    # plot LSD analysis
    spirouLSD.polar_lsd_plot(p, loc)

