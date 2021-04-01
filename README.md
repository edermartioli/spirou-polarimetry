![Alt text](Figures/SPIRou-polarimetry.png?raw=true "Title")

`SPIRou-Polarimetry` is the SPIRou DRS module to perform polarimetry calculations.

To start using `SPIRou-Polarimetry`, first make sure you have all the depencies installed. Most of them are installed by default in anaconda Python 3.X.

Then download all files in this repository and run the following example:

```
cd spirou-polarimetry/data/GamEqu/

python ../../spirou_pol.py --exp1=2329699e.fits --exp2=2329700e.fits --exp3=2329701e.fits 
--exp4=2329702e.fits --output=2329699p.fits --output_lsd=2329699_lsd.fits -p -s -L
```

Check if your results are similar to the products provided in the directory `expected_results_of_examples`, or just compare the final LSD analysis with the one below:

![Alt text](Figures/GamEqu_spirou-lsd.png?raw=true "Title")

Change input parameters in the file `polar_param.py` to test different flavors of reduction.

To process a full data set automatically for several sequences obtained at different epochs one may first link all `*e.fits` and `*t.fits` data into a given directory, for example:
```
cd $MY_PATH/OBJECT/
ln -s $PATH_TO_REDUCED_DATA/OBJECT/*e.fits .
ln -s $PATH_TO_REDUCED_DATA/OBJECT/*t.fits .
```

Then one can run the polarimetry pipeline as in the following exemple below:

```
python ~/spirou-polarimetry/spirou_pol_pipeline.py --input=*e.fits -Lsb
```
The command line above will identify all 4-exposure polarimetric sequences in the input dataset, and will calculate the polarimetric spectra `*p.fits` for every sequence. Then the options `-L -s -b` will add the following processing steps:
```
-L to calculate the least-squares deconvolution (LSD) profiles and save them as *_lsd.fits files. 
-s to stack all LSD profiles and save it as OBJECT_lsd_stack.fits 
-b to calculate the longitudinal magnetic field time series and save it as OBJECT_blong.rdb
```




