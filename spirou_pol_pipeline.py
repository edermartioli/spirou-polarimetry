# -*- coding: iso-8859-1 -*-
"""
    Created on April 29 2020
    
    Description: This routine identifies and reduce all polarimetric sequences in a given data set.
    
    @author: Eder Martioli <martioli@iap.fr>
    
    Institut d'Astrophysique de Paris, France.
    
    Simple usage example:
    
    python ~/spirou-tools/spirou-polarimetry/spirou_pol_pipeline.py --input=*e.fits -Lsb -p -v
    
    """

__version__ = "1.0"

__copyright__ = """
    Copyright (c) ...  All rights reserved.
    """

from optparse import OptionParser
import os,sys
import astropy.io.fits as fits
import glob
from copy import deepcopy


def generate_polar_sets(file_list, verbose=False) :

    polar_sets = {}
    
    current_exp_num = 0
    pol_sequence = ["","","",""]

    for i in range(len(file_list)) :

        hdr = fits.getheader(file_list[i])
    
        if "SBRHB1_P" in hdr.keys() and "SBRHB2_P" in hdr.keys() :
        
            if hdr["SBRHB1_P"] == "P16" and hdr["SBRHB2_P"] == "P16" :
                current_exp_num = 0
                if verbose :
                    print("File:",file_list[i], "is in spectroscopic mode, skipping ...")
                continue
        
            elif (hdr["SBRHB1_P"] == "P16" and hdr["SBRHB2_P"] == "P2") or \
                (hdr["SBRHB1_P"] == "P2" and hdr["SBRHB2_P"] == "P14") or \
                (hdr["SBRHB1_P"] == "P14" and hdr["SBRHB2_P"] == "P16") :
                pol_sequence[0] = file_list[i]
            
                current_exp_num = 1
        
            elif (hdr["SBRHB1_P"] == "P16" and hdr["SBRHB2_P"] == "P14") or \
                 (hdr["SBRHB1_P"] == "P2" and hdr["SBRHB2_P"] == "P2") or \
                (hdr["SBRHB1_P"] == "P2" and hdr["SBRHB2_P"] == "P16") :
                if current_exp_num == 1 :
                    if verbose :
                        print("File:",file_list[i], "is exposure 2, OK ...")
                    pol_sequence[current_exp_num] = file_list[i]
                    current_exp_num = 2
                elif current_exp_num == -1 :
                    if verbose:
                        print("File",file_list[i]," is part of skipped seqeuence ...")
                    continue
                else :
                    current_exp_num = 0
                    if verbose :
                        print("File",file_list[i]," is exposure 2, but sequence is out-of-order, skipping ...")
                    continue
                        
            elif (hdr["SBRHB1_P"] == "P4" and hdr["SBRHB2_P"] == "P2") or \
                 (hdr["SBRHB1_P"] == "P14" and hdr["SBRHB2_P"] == "P14") or \
                 (hdr["SBRHB1_P"] == "P2" and hdr["SBRHB2_P"] == "P4") :
                     
                if current_exp_num == 2 :
                    if verbose :
                        print("File:",file_list[i], "is exposure 3, OK ...")
                    pol_sequence[current_exp_num] = file_list[i]
                    current_exp_num = 3
                elif current_exp_num == -1 :
                    if verbose :
                        print("File",file_list[i]," is part of skipped seqeuence ...")
                    continue
                else :
                    current_exp_num = 0
                    if verbose :
                        print("File",file_list[i]," is exposure 3, but sequence is out-of-order, skipping ...")
                    continue

            elif (hdr["SBRHB1_P"] == "P4" and hdr["SBRHB2_P"] == "P14") or \
                 (hdr["SBRHB1_P"] == "P14" and hdr["SBRHB2_P"] == "P2") or \
                 (hdr["SBRHB1_P"] == "P14" and hdr["SBRHB2_P"] == "P4") :
                     
                if current_exp_num == 3 :
                    if verbose :
                        print("File:",file_list[i], "is exposure 4, OK ...")
                    
                    pol_sequence[current_exp_num] = file_list[i]
                
                    if verbose:
                        print("Stacking polarimetric sequence:", pol_sequence)
                
                    polar_sets[pol_sequence[0]] = deepcopy(pol_sequence)
                
                    current_exp_num = 0
                elif current_exp_num == -1 :
                    if verbose :
                        print("File",file_list[i]," is part of skipped seqeuence ...")
                    continue
                else :
                    current_exp_num = 0
                    if verbose :
                        print("File",file_list[i]," is exposure 4, but sequence is out-of-order, skipping ...")
                    continue
            else :
                current_exp_num = 0
                if verbose :
                    print("File:",file_list[i], "is in UNKNOWN mode, skipping ...")
                continue

        else :
            current_exp_num = 0
            if verbose :
                print("File:",file_list[i], "does not have keywords SBRHB1_P and SBRHB2_P, skipping ...")
            continue

    return polar_sets


def generate_polar_continuous_sets(polar_sets, verbose=False) :

    cont_polar_sets = {}

    keys = list(polar_sets.keys())

    nkeys = len(keys)
    
    for j in range(nkeys):
        set = polar_sets[keys[j]]
        
        cont_polar_sets[set[0]] = [set[0], set[1], set[2], set[3]]
        if j < nkeys - 1 :
            nset = polar_sets[keys[j+1]]
            cont_polar_sets[set[1]] = [nset[0], set[1], set[2], set[3]]
            cont_polar_sets[set[2]] = [nset[0], nset[1], set[2], set[3]]
            cont_polar_sets[set[3]] = [nset[0], nset[1], nset[2], set[3]]

    if verbose:
        for key in cont_polar_sets.keys() :
            print(key,"->",cont_polar_sets[key])

    return cont_polar_sets


parser = OptionParser()
parser.add_option("-i", "--input", dest="input", help="Input spectral e.fits data pattern",type='string',default="*e.fits")
parser.add_option("-m", "--lsdmask", dest="lsdmask", help="Input LSD mask",type='string',default="")
parser.add_option("-c", action="store_true", dest="contset", help="Produce continuous set", default=False)
parser.add_option("-L", action="store_true", dest="run_lsd", help="Run LSD analysis", default=False)
parser.add_option("-s", action="store_true", dest="stack_lsd_profiles", help="Stack LSD profiles", default=False)
parser.add_option("-b", action="store_true", dest="blong_timeseries", help="Calculate longitudinal magnetic field time series", default=False)
parser.add_option("-v", action="store_true", dest="verbose", help="verbose", default=False)
parser.add_option("-p", action="store_true", dest="plot", help="plot", default=False)

try:
    options,args = parser.parse_args(sys.argv[1:])
except:
    print("Error: check usage with  -h spirou_pol_pipeline.py")
    sys.exit(1)

if options.verbose:
    print('Spectral e.fits data pattern: ', options.input)
    print('LSD mask: ', options.lsdmask)

spirou_pol_dir = os.path.dirname(__file__) + '/'

# make list of efits data files
if options.verbose:
    print("Creating list of e.fits spectrum files...")
inputedata = sorted(glob.glob(options.input))

polar_sets = generate_polar_sets(inputedata)

if options.contset :
    polar_sets = generate_polar_continuous_sets(polar_sets, verbose=True)

object_name = "object"

for key in polar_sets.keys() :
    
    output_pol = str(key).replace("e.fits","p.fits")
    
    if object_name == "object" :
        object_name = fits.getheader(key,0)["OBJECT"].replace(" ","")
    
    seq = polar_sets[key]
    
    if options.run_lsd :
        output_lsd = str(key).replace("e.fits","_lsd.fits")
    
        command = "python {0}spirou_pol.py --exp1={1} --exp2={2} --exp3={3} --exp4={4} --lsdmask={5} --output={6} --output_lsd={7} -L".format(spirou_pol_dir,seq[0],seq[1],seq[2],seq[3],options.lsdmask, output_pol,output_lsd)
    else :
        command = "python {0}spirou_pol.py --exp1={1} --exp2={2} --exp3={3} --exp4={4} --output={5}".format(spirou_pol_dir,seq[0],seq[1],seq[2],seq[3], output_pol)

    print("Running: ",command)
    os.system(command)


if "e.fits" in options.input :
    lsd_pattern = (options.input).replace("e.fits","_lsd.fits")
else :
    lsd_pattern = "*_lsd.fits"

plot_flag = ""
if options.plot :
    plot_flag = "-p"
verbose_flag = ""
if options.verbose :
    verbose_flag = "-v"

if options.stack_lsd_profiles and options.run_lsd :
    output_stack_lsd = object_name + "_lsd_stack.fits"
    
    command = "python {0}stack_lsd_profiles.py --input={1} --output={2} {3} {4}".format(spirou_pol_dir, lsd_pattern, output_stack_lsd, plot_flag, verbose_flag)

    print("Running: ",command)
    os.system(command)

if options.blong_timeseries and options.run_lsd :
    output_blong_timeseries = object_name + "_blong.rdb"
    
    command = "python {0}spirou_blong_timeseries.py --input={1} --output={2} {3} {4}".format(spirou_pol_dir, lsd_pattern, output_blong_timeseries, plot_flag, verbose_flag)

    print("Running: ",command)
    os.system(command)
