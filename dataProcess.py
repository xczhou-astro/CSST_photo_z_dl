import tensorflow as tf
import os
import numpy as np
import warnings
from astropy.io import fits
import copy

warnings.simplefilter("ignore", UserWarning)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class DataProcess:
    
    def __init__(self, data_type, mode='train', augmentation=True,
                 aug_n=50, batch_size=256):
        
        self.data_type = data_type
        self.mode = mode
        if self.mode == 'train':
            self.augmentation = augmentation
            self.aug_n = aug_n
        else:
            self.augmentation = False
            self.aug_n = 1
            
        self.batch_size = batch_size
        
        self.sigma_filter = [10.65, 7.84, 9.94, 10.59, 10.59, 9.56, 11.53]
        
        self.__check_constructor()
        
    def to_float32(self, data):
        return data.astype(np.float32)
        
    def __check_constructor(self):
        if (self.data_type != 'photometry') & (self.data_type != 'image') \
            & (self.data_type != 'photometry_and_image'):
            print('data_type can only be photometry, image or photometry_and_image')
            
        if (self.mode != 'train') & (self.mode != 'evaluate') & (self.mode != 'inference'):
            print('mode can only be train, evaluate and inference')
    
    def random_padding(self, input_image, out_dim, sigma):

        dim = input_image.shape[0]
        channel = input_image.shape[-1]

        output_shape = (out_dim, out_dim, channel)

        container = np.random.normal(scale=sigma, size=output_shape)

        for ch in range(channel):
            for i in range(dim):
                for j in range(dim):
                    start_x = (out_dim - dim) // 2
                    start_y = (out_dim - dim) // 2

                    container[start_x + i, start_y + j, ch] = input_image[i, j, ch]

        return container
    
    def crop_center(self, img, cropx, cropy):
        y, x, c = img.shape
        startx = x//2 - (cropx//2)
        starty = y//2 - (cropy//2)
        return img[starty:starty+cropy, startx:startx+cropx, :]
    
    def image_process(self, imgname, threshold=32):
        
        file = fits.open(imgname)
        image = file[0].data
        file.close()
        
        image = self.to_float32(image)
        
        size = image.shape[1]
            
        if size > threshold:
            img_rescaled = self.crop_center(image, threshold, threshold)
        elif size < threshold:
            img_rescaled = self.random_padding(image, threshold,
                                               sigma=self.sigma_filter)
        else:
            img_rescaled = image
        
        return img_rescaled
    
    def flux_process(self, flux, err_flux):
        
        mag_max = np.array([25.4, 25.5, 26.2, 26.0, 25.8, 25.7, 25.5])
        f_nu = 10**((mag_max + 48.6) / -2.5)
        
        def func(x):
            if x > 0:
                return np.log10(x)
            else:
                return -np.log10(-x)
        
        vfunc = np.vectorize(func)
        flux_scale = vfunc(flux / f_nu)
        err_flux_scale = vfunc(np.abs(err_flux / flux))
        color_like_scale = vfunc(flux[:, :-1] / flux[:, 1:])
        
        return np.column_stack((flux_scale, color_like_scale, err_flux_scale))
        
    def load_catalogue(self, catalogue_filename):
        
        print('Read in catalogue.')
        
        self.catalogue_filename = catalogue_filename
        catalogue = fits.open(catalogue_filename)
        self.catalogue_data = catalogue[1].data
        self.catalogue_header = catalogue[1].header
        catalogue.close()
        
    def read_flux(self, catalogue_data, flux_keys, flux_error_keys):
        
        fluxes = []
        for key in flux_keys:
            fluxes.append(catalogue_data[key])

        err_fluxes = []
        for key in flux_error_keys:
            err_fluxes.append(catalogue_data[key])
            
        return fluxes, err_fluxes
    
    def load_specz(self, specz=None, specz_key=None):
        
        if specz is not None:
            
            print('Read in Spec-zs.')
            
            if specz_key is not None:
                print('specz provided, override specz_key.')
        else:
            
            print('Read in Spec-zs from catalogue.')
            
            specz = self.catalogue_data[specz_key]
            
        specz = self.to_float32(specz)
        
        ds_specz = tf.data.Dataset.from_tensor_slices(specz)
        
        self.ds_specz = ds_specz
                
    
    def load_photometry(self, flux_data=None, flux_keys=None, flux_error_keys=None):
        
        if flux_data is not None:
            
            print('Read in photometry data.')
            
            if (flux_keys is not None) | (flux_error_keys is not None):
                print('flux data provided, override flux_keys and flux_error_keys')
            fluxes, err_fluxes = flux_data
            fluxes = self.to_float32(fluxes)
            err_fluxes = self.to_float32(err_fluxes)
            
        elif (flux_data is None) & (flux_keys is not None) & (flux_error_keys is not None):
            
            print('Read in photometry data from catalogue.')
            
            fluxes, err_fluxes = self.read_flux(self.catalogue_data, 
                                                flux_keys, flux_error_keys)
            fluxes = self.to_float32(fluxes)
            err_fluxes = self.to_float32(err_fluxes)
        else:
            raise NotImplementedError
            
        flux_data = self.flux_process(fluxes, err_fluxes)
        dim = flux_data.shape[-1]
        datasize_orig = flux_data.shape[0]
        
        if not self.augmentation:
            
            datasize = datasize_orig
            
            def generator():
                for data in flux_data:
                    yield data
                    
            ds_photometry = tf.data.Dataset.from_generator(
                generator, output_signature=(
                    tf.TensorSpec(shape=(dim,), dtype=tf.float32)
                )
            )
        else:
            
            datasize = datasize_orig * self.aug_n
            
            data = np.column_stack((fluxes, err_fluxes))
            ds_photometry = self.photometry_augmentation(data)
        
        self.ds_photometry = ds_photometry
        self.datasize = datasize
        
    def photometry_augmentation(self, data):
        
        print('Perform data augmentation for photometry data.')
        
        datasize = data.shape[0]

        photo_aug = []
        for _ in range(self.aug_n - 1):
            data_temp = copy.deepcopy(data)
            data_temp[:, 0:7] = data_temp[:, 0:7] +\
                np.random.normal(0, scale=data_temp[:, 7:14])
            
            flux, err_flux = data_temp[:, 0:7], data_temp[:, 7:14]
            processed = self.flux_process(flux, err_flux)

            photo_aug.append(processed)

        data_processed = self.flux_process(data[:, 0:7], data[:, 7:14])
        dim = data_processed.shape[-1]
        
        if self.aug_n == 1:
            photo_aug = data_processed
        else:
            photo_aug = np.array(photo_aug)
            photo_aug = photo_aug.reshape((self.aug_n - 1) * datasize.shape[0], -1)
            photo_aug = np.row_stack((data_processed, photo_aug))
        
        def generator():
            for photo_data in photo_aug:
                yield photo_data
        
        ds = tf.data.Dataset.from_generator(generator,
                                            output_signature=(
                                                tf.TensorSpec(shape=(dim,), dtype=tf.float32)
                                            ))
        return ds
    
    def transform(self, arr, n):
        if n == 0:
            transform = np.rot90(arr, k=0, axes=(0, 1))
        elif n == 1:
            transform = np.rot90(arr, k=1, axes=(0, 1))
        elif n == 2:
            transform = np.rot90(arr, k=2, axes=(0, 1))
        elif n == 3:
            transform = np.flip(arr, axis=0)
        elif n == 4:
            transform = np.flip(arr, axis=1)
        elif n == 5:
            transform = np.rot90(np.flip(arr, axis=0), axes=(0, 1))
        elif n == 6:
            transform = np.rot90(np.flip(arr, axis=1), axes=(0, 1))
        else:
            transform = arr

        return transform
    
    def image_augmentation(self, datasize, images=None, imgnames=None, random_aug=False):
        
        print('Perform data augmentation for image data.')
        
        idx = np.arange(datasize)
        aug_idx = np.repeat(idx, 8)
        transform_n = np.tile(np.arange(8), datasize)

        datasize_aug = datasize * 8
        shuffle_idx = np.random.choice(datasize_aug, datasize_aug, replace=False)
        aug_idx = aug_idx[shuffle_idx]
        transform_n = transform_n[shuffle_idx]
        
        if random_aug:
            aug_idx = np.tile(np.arange(datasize), self.aug_n)
            datasize_aug = datasize * self.aug_n
            transform_n = np.random.choice(datasize_aug, 8, replace=True)
            
            shuffle_idx = np.random.choice(datasize_aug, datasize_aug, replace=False)
            
            aug_idx = aug_idx[shuffle_idx]
            transform_n = transform_n[shuffle_idx]            
            
        
        if images is not None:
            def generator():
                for i, n in zip(aug_idx, transform_n):
                    yield self.transform(images[i], n)
            
        elif imgnames is not None:
            def generator():
                for i, n in zip(aug_idx, transform_n):
                    yield self.transform(self.image_process(imgnames[i]), n)
                    
        else:
            raise NotImplementedError
        
        ds = tf.data.Dataset.from_generator(generator, 
                                            output_signature=(
                                                tf.TensorSpec(shape=(32, 32, 7), dtype=tf.float32)
                                            ))
        return ds
            
    def image_ds(self, images=None, imgnames=None):
        if images is not None:
            def generator():
                for image in images:
                    yield image
            
        elif imgnames is not None:
            def generator():
                for imgname in imgnames:
                    yield self.image_process(imgname)
        else:
            raise NotImplementedError
        
        ds = tf.data.Dataset.from_generator(generator, 
                                            output_signature=(
                                                tf.TensorSpec(shape=(32, 32, 7), dtype=tf.float32)
                                            ))
        return ds
    
    def load_images(self, images=None, imgnames=None, random_aug=False):
        if images is not None:
            
            print('Read in image data.')
            
            if imgnames is not None:
                print('images provided, override imgnames')
            
            images = self.to_float32(images)
            data = images
            datatype = {}
            datatype['images'] = data
        elif (imgnames is not None) & (images is None):
            
            print('Read in filenames for images.')
            
            data = imgnames
            datatype = {}
            datatype['imgnames'] = data
        else:
            raise NotImplementedError
        
        datasize = len(data)
        
        if not self.augmentation:
            ds_image = self.image_ds(**datatype)
            datasize = datasize
        else:
            ds_image = self.image_augmentation(datasize, **datatype,
                                               random_aug=random_aug)
            if random_aug:
                datasize = datasize * self.aug_n
            else:
                datasize = datasize * 8
        
        self.ds_image = ds_image
        self.datasize = datasize
    
    def load_test_catalogue(self, test_catalogue_filename):
        
        print('Read in test catalogue.')
        
        test_catalogue = fits.open(test_catalogue_filename)
        self.test_catalogue_data = test_catalogue[1].data
        self.test_catalogue_header = test_catalogue[1].header
        test_catalogue.close()
    
    def load_test_photometry(self, flux_data=None, flux_keys=None, flux_error_keys=None):
        if flux_data is not None:
            
            print('Read in testing photometry data.')
            
            if (flux_keys is not None) | (flux_error_keys is not None):
                print('flux data provided, override flux_keys and flux_error_keys')
            fluxes, err_fluxes = flux_data
            fluxes = self.to_float32(fluxes)
            err_fluxes = self.to_float32(err_fluxes)
        elif (flux_data is None) & (flux_keys is not None) & (flux_error_keys is not None):
            
            print('Read in testing photometry data from catalogue.')
            
            fluxes, err_fluxes = self.read_flux(self.test_catalogue_data, flux_keys, flux_error_keys)
            fluxes = self.to_float32(fluxes)
            err_fluxes = self.to_float32(err_fluxes)
        flux_data = self.flux_process(fluxes, err_fluxes)
        dim = flux_data.shape[-1]
        
        def generator():
            for data in flux_data:
                yield data
                
        test_ds_photometry = tf.data.Dataset.from_generator(
            generator, output_signature=(
                tf.TensorSpec(shape=(dim,), dtype=tf.float32)
            )
        )
        self.test_ds_photometry = test_ds_photometry
    
    def load_test_images(self, images=None, imgnames=None):
        if images is not None:
            
            print('Read in testing image data.')
            
            if imgnames is not None:
                print('images provided, override imgnames')
            images = self.to_float32(images)
            data = images
            datatype = {}
            datatype['images'] = data
        elif (imgnames is not None) & (images is None):
            
            print('Read in filenames for testing images.')
            
            data = imgnames
            datatype = {}
            datatype['imgnames'] = data
        else:
            raise NotImplementedError
        
        test_ds_image = self.image_ds(**datatype)
        self.test_ds_image = test_ds_image
    
    def load_test_specz(self, specz, specz_key=None):
        if specz is not None:
            
            print('Read in testing Spec-zs.')
            
            if specz_key is not None:
                print('specz provided, override specz_key.')
        else:
            
            print('Read in testing Spec-zs from catalogue.')
            
            specz = self.catalogue_data[specz_key]
        
        specz = self.to_float32(specz)
        
        test_ds_specz = tf.data.Dataset.from_tensor_slices(specz)
        
        self.test_ds_specz = test_ds_specz
    
    def create_tfds(self, ds, ds1=None, ds2=None, prefetch=True):
        
        if ds1 is not None: 
            ds = tf.data.Dataset.zip(
                (ds, ds1)
            )
        if ds2 is not None:
            ds = tf.data.Dataset.zip(
                (ds, ds2)
            )
        ds = ds.batch(self.batch_size)
        if prefetch:
            ds = ds.prefetch(self.batch_size)
            
        return ds
    
    def get_dataset(self):
        
        print('Create tfds.')
        
        if self.mode == 'train':
            if self.data_type == 'photometry':
                ds = self.create_tfds(self.ds_photometry, self.ds_specz)
            elif self.data_type == 'image':
                ds = self.create_tfds(self.ds_image, self.ds_specz)
            elif self.data_type == 'photometry_and_image':
                ds = self.create_tfds(self.ds_photometry, self.ds_image, self.ds_specz)
            else:
                raise NotImplementedError
            
            return ds, self.datasize
        
        if self.mode == 'evaluate':
            if self.data_type == 'photometry':
                ds = self.create_tfds(self.ds_photometry)
            elif self.data_type == 'image':
                ds = self.create_tfds(self.ds_image)
            elif self.data_type == 'photometry_and_image':
                ds = self.create_tfds(self.ds_photometry, self.ds_image)
            else:
                raise NotImplementedError
            
            return ds, self.ds_specz


        elif self.mode == 'inference':
            if self.data_type == 'photometry':
                ds = self.create_tfds(self.ds_photometry)
            elif self.data_type == 'image':
                ds = self.create_tfds(self.ds_image)
            elif self.data_type == 'photometry_and_image':
                ds = self.create_tfds(self.ds_photometry, self.ds_image)
            
            return ds, self.datasize
                
        else:
            raise NotImplementedError
        
    
    def get_test_dataset(self):
        
        print('Create tfds for testing.')
        
        if self.data_type == 'photometry':
            ds = self.create_tfds(self.test_ds_photometry, prefetch=False)
        elif self.data_type == 'image':
            ds = self.create_tfds(self.test_ds_image, prefetch=False)
        elif self.data_type == 'photometry_and_image':
            ds = self.create_tfds(self.test_ds_photometry, self.test_ds_image, 
                                  prefetch=False)
        else:
            raise NotImplementedError
        
        return ds, self.test_ds_specz
    