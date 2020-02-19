![Alt text](Figures/SPIRou-polarimetry.png?raw=true "Title")

`SPIRou-Polarimetry` is the SPIRou DRS module to perform polarimetry calculations.

To start using `SPIRou-Polarimetry`, first make sure you have all the following depencies installed:

`numpy`, `scipy`, `astropy`, `matplotlib`, `optparse`, `copy`

Another option is to install python 3 via anaconda.

Then download all files in this repository and run the following example:

```
cd spirou-polarimetry/data/GamEqu/

python ../../spirou_pol.py --exp1=2329699e.fits --exp2=2329700e.fits --exp3=2329701e.fits 
--exp4=2329702e.fits --lsdmask=../../lsd_masks/marcs_t5000g50_all --output=2329699_pol.fits 
--output_lsd=2329699_lsd.fits -p -s -L
```

Check if your results are similar to the products provided in the directory `expected_results_of_examples`, or just compare the final LSD analysis with the one below:

![Alt text](Figures/GamEqu_spirou-lsd.png?raw=true "Title")

The user can change input parameters in the file `polar_param.py` to test different flavors of reduction.


