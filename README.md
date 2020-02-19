![Alt text](Figures/SPIRou-polarimetry.png?raw=true "Title")

`SPIRou-polarimetry` is the spectro-polarimetry module to process the SPIRou DRS.

To start using the SPIRou-Polarimetry, first make sure you have the following depencies installed:

`numpy`, `scipy`, `astropy`, `matplotlib`, `optparse`, `copy`

Then download all files in this repository and run the following example:

```
cd spirou-polarimetry/data/GamEqu/

python ../../spirou_pol.py --exp1=2329699e.fits --exp2=2329700e.fits --exp3=2329701e.fits --exp4=2329702e.fits --lsdmask=../../lsd_masks/marcs_t5000g50_all --output=2329699_pol.fits --output_lsd=2329699_lsd.fits -p -s -L
```

Check if your results are similar to the products provided in the directory `expected_results_of_examples`.
