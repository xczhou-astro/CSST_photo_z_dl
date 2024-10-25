# CSST_photo_z_dl
Public code for estimating photo-z by CNN and BNN for CSST

For further details on methodologies and architectures, please consult our papers:

[Paper 1](https://doi.org/10.1093/mnras/stac786): "Extracting Photometric Redshift from Galaxy Flux and Image Data using Neural Networks in the CSST Survey"  
and   
[Paper 2](https://doi.org/10.1088/1674-4527/ac9578): "Photometric redshift estimates using Bayesian neural networks in the CSST survey"  

## Dependence
Required python version:  
`python==3.9.20`

Required python packages:  
`numpy==1.26.0`  
`scipy==1.13.1`  
`astropy==6.0.1`  
`matplotlib==3.9.2`  
`tensorflow==2.14.0`  
`tensorflow-probability==0.22.1`  
`tqdm==4.66.5`  

A new conda enviroment is recommended to run our code:  
`conda create -n photz_csst python=3.9.20`  
`conda activate photz_csst`  
and install dependence:  
`pip install <package_name>`

## Usage

### Imports
```Python
from dataProcess import DataProcess
from photzEstimator import PhotzEstimator
```  

### Constructor for DataProcess 
```Python
dataloader = DataProcess(data_type, mode='train', augmentation=True, aug_n=50, batch_size=256)
```  
`data_type`: `str`: `'photometry'`, `'image'` or `'photometry_and_image'`, input data type  
`mode`: `str`: `'train'`, `'evaluate'` or `'inference'`, working mode  
`augmentation`: `bool`, Only work when `mode='train'`, if use data augmentation  
`aug_n` : `int`: Only work when `mode='train'`, num augments for photometry data  
`batch_size`: `int`, batch_size for tf dataset  

### Class methods for DataProcess
```Python
dataloader.load_catalogue(catalogue_filename)
```  
Load catalogue  
`catalogue_filename`: `str`: catalogue filename  

```Python
dataloader.load_photometry(flux_data=None, flux_keys=None, flux_error_keys=None)
```  
Load photometry data  
`flux_data`: `list`: `[flux, err_flux]`, flux and err flux provided  
`flux_keys`: `list`: keys for 7 band fluxes in catalogue  
`flux_error_keys`: `list`: keys for 7 band flux errors in catalogue  
Notes:  
`flux_data` overrides `flux_data` and `flux_error_keys`  

```Python
dataloader.load_images(images=None, imgnames=None)
```  
Load image data  
`images`: `numpy.array`: images provided in `(num, 32, 32, 7)`  
`imgnames`: `list`: a list of filenames for images in fits  
Notes:  
1. `images` overrides `imgnames`  
2. Images will be regularized to shape `(32, 32, 7)` if imgnames are provided, thus cost more time.  

```Python
dataloader.load_specz(specz=None, specz_key=None)
```  
Load spec-z  
`specz`: `numpy.array`: specz data in `(num,)`  
`specz_key`: `str`: key for specz in catalogue  
Notes:   
1. `specz` overrides `specz_key`  
2. Call this function when `mode=='train'` or `'evaluate'`  

If one wants to monitor results for testing data when training: 
```Python
dataloader.load_test_catalogue(test_catalogue_filename)
dataloader.load_test_images(images=None, imgnames=None)
dataloader.load_test_photometry(flux_data=None, flux_keys=None, flux_error_keys=None)
dataloader.load_test_specz(specz=None, specz_key=None)
```

```Python
loaded_data = dataloader.get_dataset()
```  
Get tf dataset and other information for photo-z estimation  
`tfds, datasize = loaded_data` if `mode='train'` or `'inference'`  
`tfds, tfds_specz = loaded_data` if `mode='evaluate'`  

```Python
loaded_test_data = dataloader.get_test_dataset()
```  
Get testing tf dataset and other information for photo-z estimation  
`test_tfds, test_tfds_specz = loaded_data` if `mode='train'`

### Constructor for PhotzEstimator
```Python
estimator = PhotzEstimator(model_type, data_type, transfer=False, outDir='outputs')
```  

`model_type`: `str`: `'NN'` or `'BNN'`, model type  
`data_type`: `str`: `'photometry'`, `'image'` or `'photometry_and_image'`, input data type  
`transfer`: `bool`: Only works when `datatype='photometry_and_image'`, if use transfer learning  
`outDir`: `bool`: output directory  

### Class methods for PhotzEstimator

```Python
estimator.get_model(datasize=50000, weights=None, cnn_weights=None, mlp_weights=None, alpha_file=None)
```  
Get model  
`datasize`: `int`: datasize for loaded data, must be provided when `model_type=BNN`  
`weights`: `str`: weights file in `.h5`  
`cnn_weights`: `str`: cnn weights file in `.h5`, only provide when `data_type='photometry_and_image'` and `transfer=True`  
`mlp_weights`: `str`: mlp weights file in `.h5`, only provide when `data_type='photometry_and_image'` and `transfer=True`  
`alpha_file`: `str`: file for calibration parameters, only provide when `model_type=BNN` and `mode='inference'`  

```Python
estimator.train(train_ds, test_ds=None, learning_rate=2e-4, epochs=200)
```  
Perform training, model weights are saved in `outDir` when training finishes.  
`train_ds`: `tf.data.Dataset`: training tf dataset loaded  
`test_ds`: `tf.data.Dataset`: testing tf dataset loaded  
`learning_rate`: `float`: learning_rate for Adam optimizer  
`epochs`: `int`: number of epochs  

```Python
estimator.evaluate(ds, ds_specz, n_runs=200)
```
Perform evaluation, plot for results and results file are saved in `outDir` when evaluation finishes.  
`ds`: `tf.data.Dataset`: tf dataset for evaluation  
`ds_specz`: `tf.data.Dataset`: tf dataset for spec-z for evaluation  
`n_runs`: `int`: number of runs for `ds` to BNN model  

```Python
estimator.inference(ds, datasize, catalogue=None, info_keys=['ra', 'dec'], n_runs=200)
```  
Perfrom inference, photo-z catalogue is saved in `outDir` when inference finishes.  
`ds`: `tf.data.Dataset`: tf dataset for inference  
`datasize`: `int`: data size for `ds`  
`catalogue`: `str`: catalogue filename   
`info_keys`: `list`: keys for catalogue  
`n_runs`: `int`: number of runs for `ds` to BNN model  
Notes:  
if `catalogue=None`, a catalogue of photo-z prediction will be created, otherwise, a new catalogue of photo-z prediction including information indicated by `info_keys` will be created.  

## Example

1. If one wants to train a BNN model for data of photometry and images, and check the performance for testing data:  
```Python
from dataProcess import DataProcess
from photzEstimator import PhotzEstimator

dataloader = DataProcess(data_type, mode='train', augmentation=True, aug_n=50, batch_size=1024)
dataloader.load_catalogue('catalogue_filename')

bands = ['NUV', 'u', 'g', 'r', 'i', 'z', 'y']
flux_keys = [f'flux_{bd}' for bd in bands]
flux_error_keys = [f'fluxerr_{bd}' for bd in bands]
dataloader.load_photometry(flux_keys=flux_keys, flux_error_keys=flux_error_keys)

imgnames = ['stamp_000.fits', 'stamp_001.fits', 'stamp_002.fits' '..']
dataloader.load_images(imgnames=imgnames)

dataloader.load_specz(specz_key=['zspec'])

dataloader.load_test_catalogue('test_catalogue_filename')

images = np.load('image_arrays.npy')
dataloader.load_test_images(images=images)

fluxes = np.load('flux_data.npy')
err_fluxes = np.load('err_flux_data.npy')
dataloader.load_test_photometry(flux_data=[fluxes, err_fluxes])

specz = np.loadtxt('zspec.txt')
dataloader.load_test_specz(specz=specz)

loaded_data = dataloader.get_dataset()

tfds, datasize = loaded_data

loaded_test_data = dataloader.get_test_dataset()

test_tfds, test_tfds_specz = loaded_test_data

estimator = PhotzEstimator(model_type='BNN', data_type='photometry_and_image', transfer=False, outDir='outputs')

estimator.get_model(datasize=datasize)

estimator.train(train_dstfds, test_ds=test_tfds, learning_rate=2e-4, epochs=200)

estimator.evaluate(test_tfds, test_tfds_specz, n_runs=200)
```  
Data products:  
weights file `BNN_models/Hybrid/Hybrid_weights.h5`  
calibration parameter file `BNN_models/alpha.json`  
figure `BNN_models/Hybrid/loss.png`  
figure `BNN_models/Hybrid/acc.png`  
figure `BNN_models/Hybrid/results.png`  

2. If one wants to inference from image data using trained BNN model
```Python
from dataProcess import DataProcess
from photzEstimator import PhotzEstimator

dataloader = DataProcess(data_type, mode='inference', augmentation=True, aug_n=50, batch_size=1024)
dataloader.load_catalogue('catalogue_filename')

imgnames = ['stamp_000.fits', 'stamp_001.fits', 'stamp_002.fits' '..']
dataloader.load_images(imgnames=imgnames)

loaded_data = dataloader.get_dataset()

tfds, datasize = loaded_data

estimator = PhotzEstimator(model_type='BNN', data_type='image', transfer=False, outDir='outputs')

estimator.get_model(weights='Data/BNN/CNN_BNN_weights.h5', alpha_file='Data/BNN/alpha.json')

estimator.inference(ds=tfds, datasize=datasize, catalogue=dataloader.catalogue_filename, info_keys=['ra', 'dec'], n_runs=200)
```  
Data products:  
catalogue file `BNN_models/CNN/photoz_catalogue.fits` including `ra, dec, z_pred, z_err`  