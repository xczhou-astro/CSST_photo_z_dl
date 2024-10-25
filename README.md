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
`augmentation`: `bool`, Only work when `mode=='train'`, if use data augmentation
`aug_n` : `int`: Only work when `mode=='train'`, num augments for photometry data
`batch_size`: `int`, batch_size for tf dataset

### Class methods for DataProcess

`dataloader.load_catalogue(catalogue_filename)`
`catalogue_filename`: `str`: catalogue filename

`dataloader.load_photometry(flux_data=None, flux_keys=None, flux_error_keys=None)`
`flux_data`: `list`: `[flux, err_flux]`, flux and err flux provided
`flux_keys`: `list`: keys for 7 band fluxes in catalogue
`flux_error_keys`: `list`: keys for 7 band flux errors in catalogue
Notes: 
`flux_data` overrides `flux_data` and `flux_error_keys`

`dataloader.load_images(images=None, imgnames=None)`
`images`: `numpy.array`: images provided in `(num, 32, 32, 7)`
`imgnames`: `list`: a list of filenames for images in fits
Notes:
1. `images` overrides `imgnames`
2. Images will be regularized if imgnames are provided, thus cost more time.

`dataloader.load_specz(specz, specz_key)`
`specz`: `numpy.array`: specz data in `(num,)`
`specz_key`: `str`: key for specz in catalogue
Notes: 
1. `specz` overrides `specz_key`
2. Call this function when `mode=='train'` or `'evaluate'`

If one wants to monitor results for testing data when training:
`dataloader.load_test_catalogue(test_catalogue_filename)`
`dataloader.load_test_images(images=None, imgnames=None)`
`dataloader.load_test_photometry(flux_data=None, flux_keys=None, flux_error_keys=None)`
`dataloader.load_test_specz(specz=None, specz_key=None)`

`loaded_data = dataloader.get_dataset()`
Get tf dataset and other information for photo-z estimation
`tfds, datasize = loaded_data` if `mode='train'` or `'inference'`
`tfds, tfds_specz = loaded_data` if `mode='evaluate'`

`loaded_test_data = dataloader.get_test_dataset()`
Get testing tf dataset and other information for photo-z estimation
`test_tfds, test_datasize = loaded_data` if `mode='train'` or `mode='evaluate'`
`test_tfds, test_tfds_specz = loaded_data` if `mode='train'` or `mode='evaluate'`

### Constructor for PhotzEstimator
```Python
estimator = PhotzEstimator(model_type, data_type, transfer=False, outDir='outputs')
```

`model_type`: `str`: `'NN'` or `'BNN'`, model type
`data_type`: `str`: `'photometry'`, `'image'` or `'photometry_and_image'`, input data type
`transfer`: `bool`: Only works when `datatype='photometry_and_image'`, if use transfer learning
`outDir`: `bool`: output directory

### Class methods for PhotzEstimator

`estimator.get_model(datasize=50000, weights=None, cnn_weights=None, mlp_weights=None, alpha_file=None)`
`datasize`: `int`: datasize for loaded data, must be provided when `model_type=BNN`
`weights`: `str`: weights file in `.h5`
`cnn_weights`: `str`: cnn weights file in `.h5`, only provide when `data_type='photometry_and_image'` and `transfer=True`
`mlp_weights`: mlp weights file in `.h5`, only provide when `data_type='photometry_and_image'` and `transfer=True`
`alpha_file`: `str`: file for calibration parameters, only provide when `model_type=BNN` and `mode='inference'`

`estimator.train(train_ds, test_ds=None, learning_rate=2e-4, epochs=200)`
`train_ds`: `tf.data.Dataset`: training tf dataset loaded
`test_ds`: `tf.data.Dataset`: testing tf dataset loaded
`learning_rate`: `float`: learning_rate for Adam optimizer
`epochs`: `int`: number of epochs

`estimator.evaluate(ds, ds_specz, n_runs=200)`
`ds`: `tf.data.Dataset`: tf dataset for evaluation
`ds_specz`: `tf.data.Dataset`: tf dataset for spec-z for evaluation
`n_runs`: `int`: number of runs for `ds` to BNN model

`estimator.inference(ds, datasize, catalogue=None, info_keys=['ra', 'dec'], n_runs=200)`
`ds`: `tf.data.Dataset`: tf dataset for inference
`datasize`: `int`: data size for `ds`
`catalogue`: `str`: catalogue filename 
`info_keys`: `list`: keys for catalogue
`n_runs`: `int`: number of runs for `ds` to BNN model
Notes:
if `catalogue=None`, a catalogue of photo-z prediction will be created, otherwise, a new catalogue of photo-z prediction including information indicated by `info_keys` will be created.

