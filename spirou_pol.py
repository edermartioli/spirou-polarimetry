# -*- coding: iso-8859-1 -*-
"""
    Created on December 2 2019
    
    Description: This routine performs calculation of the polarimetric spectrum.
    
    @author: Eder Martioli <martioli@iap.fr>
    
    Institut d'Astrophysique de Paris, France.
    
    Simple usage example:
    
    1) output Libre-esprit style:
    
    python $PATH/spirou_pol.py --exp1=2329699e.fits --exp2=2329700e.fits --exp3=2329701e.fits --exp4=2329702e.fits --output=2329699.s -p
    
    2) output FITS SPIRou DRS style:
    
    2.1) without LSD analysis
    python $PATH/spirou_pol.py --exp1=2329699e.fits --exp2=2329700e.fits --exp3=2329701e.fits --exp4=2329702e.fits --output=2329699_pol.fits
    
    2.2) with LSD analysis
    python $PATH/spirou_pol.py --exp1=2329699e.fits --exp2=2329700e.fits --exp3=2329701e.fits --exp4=2329702e.fits --output=2329699_pol.fits --output_lsd=2329699_lsd.fits -p -s -L
    
    option -L activates LSD analysis
    
    option --lsdmask provides an specific mask for LSD analysis
    e.g.: --lsdmask=$PATH/lsd_masks/marcs_t5000g50_all
    
    option -p activates plotting for polarimetry
    
    option -s activates plotting for LSD analysis

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
parser.add_option("-1", "--exp1", dest="exp1", help="Input exposure 1",type='string',default="")
parser.add_option("-2", "--exp2", dest="exp2", help="Input exposure 2",type='string',default="")
parser.add_option("-3", "--exp3", dest="exp3", help="Input exposure 3",type='string',default="")
parser.add_option("-4", "--exp4", dest="exp4", help="Input exposure 4",type='string',default="")
parser.add_option("-m", "--lsdmask", dest="lsdmask", help="LSD mask",type='string', default="")
parser.add_option("-o", "--output", dest="output", help="Output file",type='string', default="")
parser.add_option("-l", "--output_lsd", dest="output_lsd", help="Output LSD file",type='string', default="")
parser.add_option("-p", action="store_true", dest="plot", help="plot", default=False)
parser.add_option("-s", action="store_true", dest="plot_lsd", help="plot_lsd", default=False)
parser.add_option("-L", action="store_true", dest="lsd", help="Run LSD analysis", default=False)
parser.add_option("-v", action="store_true", dest="verbose", help="verbose", default=False)

try:
    options,args = parser.parse_args(sys.argv[1:])
except:
    print("Error: check usage with  -h spirou_pol.py")
    sys.exit(1)

if options.verbose:
    print('Input exposure 1: ', options.exp1)
    print('Input exposure 2: ', options.exp2)
    print('Input exposure 3: ', options.exp3)
    print('Input exposure 4: ', options.exp4)
    print('LSD mask: ', options.lsdmask)
    print('Output file: ', options.output)
    print('Output LSD file: ', options.output_lsd)

# define polarimetry parameters
p = polar_param.load_polar_parameters()

p['INPUT_FILES'] = [options.exp1, options.exp2, options.exp3, options.exp4]

# set up data storage
polardict = spirouPolar.sort_polar_files(p)

# set up data storage
loc = {}

# load files
p, loc = spirouPolar.load_data(p, polardict, loc)

# ----------------------------------------------------------------------
# Polarimetry computation
# ----------------------------------------------------------------------
loc = spirouPolar.calculate_polarimetry(p, loc)

# ----------------------------------------------------------------------
# Stokes I computation
# ----------------------------------------------------------------------
loc = spirouPolar.calculate_stokes_i(p, loc)

# ----------------------------------------------------------------------
# Calculate continuum (for plotting)
# ----------------------------------------------------------------------
loc = spirouPolar.calculate_continuum(p, loc)

# ----------------------------------------------------------------------
# Remove continuum polarization
# ----------------------------------------------------------------------
if p['IC_POLAR_REMOVE_CONTINUUM'] :
    loc = spirouPolar.remove_continuum_polarization(loc)

# ----------------------------------------------------------------------
# Normalize Stokes I
# ----------------------------------------------------------------------
if p['IC_POLAR_NORMALIZE_STOKES_I'] :
    loc = spirouPolar.normalize_stokes_i(loc)

# ----------------------------------------------------------------------
# Apply sigma-clipping
# ----------------------------------------------------------------------
if p['IC_POLAR_CLEAN_BY_SIGMA_CLIPPING'] :
    loc = spirouPolar.clean_polarimetry_data(loc, sigclip=True,
        nsig=p['IC_POLAR_NSIGMA_CLIPPING'], overwrite=True)

# ----------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------
if options.plot :
    # plot continuum plots
    spirouPolar.polar_continuum_plot(p, loc)
    # plot polarimetry results
    spirouPolar.polar_result_plot(p, loc)
    # plot total flux (Stokes I)
    spirouPolar.polar_stokes_i_plot(p, loc)

if options.output.endswith(".fits") :
    spirouPolar.save_pol_fits(options.output, p, loc)
elif options.output.endswith(".s") :
    spirouPolar.save_pol_le_format(options.output, loc)

# ------------------------------------------------------------------
if options.lsd :
    
    if options.lsdmask != "" :
        # set lsd mask file from input
        loc['LSD_MASK_FILE'] = options.lsdmask
    else :
        # select an lsd mask file from repositories
        loc['LSD_MASK_FILE'] = spirouLSD.select_lsd_mask(p)

    # ------------------------------------------------------------------
    # LSD Analysis
    # ------------------------------------------------------------------
    loc = spirouLSD.lsd_analysis_wrapper(p, loc)
    
    # save LSD data to fits
    if options.output_lsd != "" :
        spirouLSD.save_lsd_fits(options.output_lsd, loc, p)

    if options.plot_lsd :
        # plot LSD analysis
        spirouLSD.polar_lsd_plot(p, loc)
