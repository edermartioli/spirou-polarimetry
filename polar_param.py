# -----------------------------------------------------------------------------
#   define polarimetry parameters
# -----------------------------------------------------------------------------
def load_polar_parameters() :
    
    #initialize parameters dictionary
    p = {}
    
    # Whether or not to use telluric subctracted flux
    p['IC_POLAR_USE_TELLURIC_CORRECTED_FLUX'] = True

    # Wheter or not to correct for BERV shift before calculate polarimetry
    p['IC_POLAR_BERV_CORRECT'] = True

    # Wheter or not to correct for SOURCE RV shift before calculate polarimetry
    p['IC_POLAR_SOURCERV_CORRECT'] = False
    
    # Normalize Stokes I (True = 1, False = 0)
    p['IC_POLAR_NORMALIZE_STOKES_I'] = True
    
    # Wheter or not to inerpolate flux values to correct for wavelength shifts between exposures
    p['IC_POLAR_INTERPOLATE_FLUX'] = True

    #  Define all possible stokes parameters                          - [pol_spirou]
    p['IC_POLAR_STOKES_PARAMS'] = ['V', 'Q', 'U']

    #  Define all possible fibers used for polarimetry                - [pol_spirou]
    p['IC_POLAR_FIBERS'] = ['A', 'B']

    #  Define the polarimetry method                                  - [pol_spirou]
    #    currently must be either:
    #         - Ratio
    #         - Difference
    p['IC_POLAR_METHOD'] = 'Ratio'

    # Apply polarimetric sigma-clip cleanning (Works better if continuum is removed)
    p['IC_POLAR_CLEAN_BY_SIGMA_CLIPPING'] = True
    
    # Define number of sigmas within which apply clipping
    p['IC_POLAR_NSIGMA_CLIPPING'] = 4
    
    #  Perform LSD analysis (True = 1, False = 0)                     - [pol_spirou]
    #p['IC_POLAR_LSD_ANALYSIS'] = 1

    #  Define initial velocity (km/s) for output LSD profile          - [pol_spirou]
    p['IC_POLAR_LSD_V0'] = -150.

    #  Define final velocity (km/s) for output LSD profile            - [pol_spirou]
    p['IC_POLAR_LSD_VF'] = 150.

    #  Define number of points for output LSD profile                 - [pol_spirou]
    p['IC_POLAR_LSD_NP'] = 151

    #  Remove edges of LSD profile                 - [pol_spirou]
    p['IC_POLAR_LSD_REMOVE_EDGES'] = True
    
    #  Define repositories of line masks for LSD analysis              - [pol_spirou]
    p['IC_POLAR_LSD_MASKS_REPOSITORIES'] = ['lsd_masks/marcs*_all']
    # Below is an example on how to include more than one repository
    #p['IC_POLAR_LSD_MASKS_REPOSITORIES'] = ['lsd_masks/*_all', 'lsd_masks/20200304-mask_nIR_EMS/masks/min-depth-0_01/t*00_g4.0_m0.00']

    #  If mask lines are in air-wavelength then they will have to be converted from air to vaccuum:
    p['IC_POLAR_LSD_CCFLINES_AIR_WAVE'] = False
    
    #  Define mask for selecting lines to be used in the LSD analysis - [pol_spirou]
    # noinspection PyPep8
    p['IC_POLAR_LSD_WLRANGES'] = [[983., 1116.], [1163., 1260.], [1280., 1331.], [1490., 1790.], [1975., 1995.], [2030., 2047.5]]

    #  Define minimum line depth to be used in the LSD analyis        - [pol_spirou]
    p['IC_POLAR_LSD_MIN_LINEDEPTH'] = 0.005
    
    #  Define minimum lande of lines to be used in the LSD analyis        - [pol_spirou]
    p['IC_POLAR_LSD_MIN_LANDE'] = 0.0
    
    #  Define maximum lande of lines to be used in the LSD analyis        - [pol_spirou]
    p['IC_POLAR_LSD_MAX_LANDE'] = 10.

    # Renormalize data before LSD analysis
    p['IC_POLAR_LSD_NORMALIZE'] = False
    
    # Renormalize data before LSD analysis
    p['IC_POLAR_LSD_NORMALIZE'] = False
    
    
    ##### Select polarization continuum detection algorithm: 'IRAF' or 'MOVING_MEDIAN'
    p['IC_POLAR_CONTINUUM_DETECTION_ALGORITHM'] = 'MOVING_MEDIAN'
    
    ##### Select stokes I continuum detection algorithm: 'IRAF' or 'MOVING_MEDIAN'
    p['IC_STOKESI_CONTINUUM_DETECTION_ALGORITHM'] = 'MOVING_MEDIAN'

    #### Parameters below are for 'MOVING_MEDIAN' algorithm :
    #  Fit polynomial to continuum polarization? (True = 1, False = 0) - [pol_spirou]
    # If False it will use a cubic interpolation instead of polynomial fit
    p['IC_POLAR_CONT_POLYNOMIAL_FIT'] = True
    #  Define degree of polynomial to fit continuum polarization      - [pol_spirou]
    p['IC_POLAR_CONT_DEG_POLYNOMIAL'] = 3
    #  Define the polarimetry continuum bin size                       - [pol_spirou]
    p['IC_POLAR_CONT_BINSIZE'] = 900
    #  Define the polarimetry continuum overlap size                   - [pol_spirou]
    p['IC_POLAR_CONT_OVERLAP'] = 200
    #  Define the telluric mask for calculation of continnum          - [pol_spirou]
    # noinspection PyPep8
    p['IC_POLAR_CONT_TELLMASK'] = [[930, 967], [1109, 1167], [1326, 1491], [1782, 1979], [1997, 2027], [2047, 2076]]
    #################################
    
    #### Parameters below are for 'IRAF' algorithm :
    #function to fit to the polar continuum: must be 'polynomial' or 'spline3'
    if p['IC_POLAR_CONT_POLYNOMIAL_FIT'] :
        p['IC_POLAR_IRAF_CONT_FIT_FUNCTION'] = "polynomial"
    else :
        p['IC_POLAR_IRAF_CONT_FIT_FUNCTION'] = "spline3"
    #function to fit to the stokes I continuum: must be 'polynomial' or 'spline3'
    p['IC_STOKESI_IRAF_CONT_FIT_FUNCTION'] = "polynomial"
    # polar continuum fit function order: 'polynomial': degree or 'spline3': number of knots
    p['IC_POLAR_IRAF_CONT_FUNCTION_ORDER'] = p['IC_POLAR_CONT_DEG_POLYNOMIAL']
    # stokes I continuum fit function order: 'polynomial': degree or 'spline3': number of knots
    p['IC_STOKESI_IRAF_CONT_FUNCTION_ORDER'] = 5
    # plot Polarimetry continuum calculation using IRAF algorithm
    p['IC_POLAR_IRAF_CONT_PLOT'] = False
    # plot Stokes I continuum calculation using IRAF algorithm
    p['IC_STOKESI_IRAF_CONT_PLOT'] = False
    #################################

    # Remove continuum polarization (True = 1, False = 0)
    p['IC_POLAR_REMOVE_CONTINUUM'] = True
    #-------------------------------------

    return p
