import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from astropy.io import fits
from astropy.table import Table
from tqdm import tqdm
import gc
import warnings
import json
from scripts.calibration import beta_calibration

warnings.simplefilter("ignore", UserWarning)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

gpu_indices = [0, 1, 5, 6]

physical_devices = tf.config.list_physical_devices('GPU')

if physical_devices:
    visible_devices = [physical_devices[i] for i in gpu_indices]
    tf.config.experimental.set_visible_devices(visible_devices, 'GPU')

    for device in visible_devices:
        tf.config.experimental.set_memory_growth(device, True)
        
class PhotzEstimator:
    
    def __init__(self, model_type, data_type,
                 transfer=False, outDir='outputs'):
        
        self.model_type = model_type
        self.data_type = data_type
        self.transfer = transfer
        self.outDir = outDir
        
        model_name = {'photometry': 'MLP', 'image': 'CNN', 'photometry_and_image': 'Hybrid'}
        self.model_name = model_name[data_type]
        
        if (self.model_name == 'photometry_and_image') & transfer:
            self.model_name += '_transfer'
            
        model_dir = os.path.join(self.outDir, f'{self.model_type}_models')
        self.data_dir = os.path.join(model_dir, f'{self.model_name}')
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.__check_constructor()
        
        self.strategy = tf.distribute.MirroredStrategy()
        
    def __check_constructor(self):
        if (self.data_type != 'photometry') & (self.data_type != 'image') \
            & (self.data_type != 'photometry_and_image'):
            print('data_type can only be photometry, image or photometry_and_image.')
            
        if (self.model_type != 'NN') & (self.model_type != 'BNN'):
            print('model_type can only be NN or BNN.')
        
    def get_model(self, datasize=50000, weights=None, 
                  cnn_weights=None, mlp_weights=None, alpha_file=None):
        
        print(f'Get {self.model_type} for {self.data_type}.')
        
        if self.model_type == 'NN':
            from scripts.model import MLP, inception, hybrid_network, hybrid_transfer_network
        elif self.model_type == 'BNN':
            from scripts.model_BNN import MLP, inception, hybrid_network, hybrid_transfer_network
        
        with self.strategy.scope():
            
            if self.model_name == 'MLP':
                self.model = MLP(datasize)
            elif self.model_name == 'CNN':
                self.model = inception(datasize)
            elif self.model_name == 'Hybrid':
                self.model = hybrid_network(datasize)
            elif self.model == 'Hybrid_transfer':
                mlp = MLP(datasize)
                mlp.load_weights(mlp_weights)
                
                cnn = inception(datasize)
                cnn.load_weights(cnn_weights)
                
                self.model = hybrid_transfer_network(mlp, cnn, datasize)
                
            if weights is not None:
                
                print('Load existing weights.')
                
                self.model.load_weights(weights)
                
                if alpha_file is not None:
                    
                    print('Read calibration parameter for existing BNN model.')
                    
                    with open(alpha_file, 'r') as file:
                        alpha = json.load(file)
                        self.alpha = alpha[self.model_name]
        
    def loss_func_CNN(self, y_true, y_pred):
        return tf.keras.losses.mean_absolute_error(y_pred=y_pred, 
                                                y_true=y_true)

    def loss_func_BNN(self, y, rv_y):
        return -rv_y.log_prob(y)
    
    def myacc(self, y_true, y_pred):
        delta = tf.math.abs(y_pred - y_true) / (1 + y_true)
        return tf.reduce_mean(tf.cast(delta <= 0.15, tf.float32), axis=-1)
    
    def train(self, train_ds, test_ds=None, 
              learning_rate=2e-4, epochs=200):
        
        print(f'Train for {self.data_type}.')
        
        loss_func = {'NN': self.loss_func_CNN,
                     'BNN': self.loss_func_BNN}
        
        with self.strategy.scope():
            self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), 
                            loss=loss_func[self.model_type],
                            metrics=[self.myacc])
        
        model_filename = f'{self.model_name}_weights.h5'
        model_filename = os.path.join(self.data_dir, model_filename)
        
        if test_ds is not None:
            cbk = tf.keras.callbacks.ModelCheckpoint(model_filename,
                                                    monitor='val_loss',
                                                    save_best_only=True, 
                                                    save_weights_only=True)
            cbk = [cbk]
        else:
            cbk = None
            
        print(f'Begin training.')
        his = self.model.fit(train_ds, epochs=epochs, 
                             validation_data=test_ds, callbacks=cbk)
        
        if cbk is None:
            self.model.save_weights(model_filename)
        
        self.model.load_weights(model_filename)
        
        self.plot_his(his, self.data_dir)
        
    def plot_his(self, his, savedir):
        keys = his.history.keys
        plt.figure(figsize=(8, 6))
        plt.plot(his.history['loss'])
        if 'val_loss' in keys:
            plt.plot(his.history['val_loss'])
        plt.savefig(os.path.join(savedir, 'loss.png'))
        
        plt.figure(figsize=(8, 6))
        plt.plot(his.history['myacc'])
        if 'val_myacc' in keys:
            plt.plot(his.history['val_myacc'])
        plt.savefig(os.path.join(savedir, 'acc.png'))
    
    
    def get_label(self, ds):
        z_true = []
        for batch_z in ds:
            z_true.append(batch_z.numpy())
        
        z_true = np.concatenate(z_true)
        return z_true
    
    def update_json_file(self, file_path, new_data):
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                data = json.load(file)
        else:
            data = {}
            
        data.update(new_data)
        
        with open(file_path, 'w') as file:
            json.dump(data, file)
        
    def evaluate(self, ds, ds_specz, n_runs=200):
        
        print(f'Evaluate {self.model_name} in {self.model_type} framework.')
        
        z_true = self.get_label(ds_specz)

        if self.model_type == 'NN':

            z_pred = self.model.predict(ds)
            z_pred = z_pred.reshape(-1)
            self.plot_result(self, z_true, z_pred, savedir=self.data_dir)
            np.savez_compressed(os.path.join(self.data_dir, 'result.npz'),
                                z_true=z_true, z_pred=z_pred)

        elif self.model_type == 'BNN':
            
            z_pred_n_runs = np.zeros((n_runs, z_true.shape[0]))

            for i in tqdm(range(n_runs)):
                z_pred_n_runs[i, :] = np.reshape(self.model.predict(ds, verbose=0),
                                                 z_true.shape[0])
                tf.keras.backend.clear_session()
                gc.collect()

            z_pred_avg = np.mean(z_pred_n_runs, axis=0)
            z_pred_std = np.std(z_pred_n_runs, axis=0)

            alpha = beta_calibration(z_true, z_pred_avg, z_pred_std)
            
            if not os.path.exists('BNN_models'):
                os.makedirs('BNN_models', exist_ok=True)
            alpha_file = os.path.join('BNN_models', 'alpha.json')
            alpha = np.around(alpha, 3)
            self.alpha = alpha
            new_data = {f'{self.model_name}': f'{alpha}'}
            self.update_json_file(alpha_file, new_data)
            
            z_pred_std_cal = z_pred_std * alpha
            self.plot_result(z_true, z_pred, z_err=z_pred_std_cal, savedir=self.data_dir)
            np.savez_compressed(os.path.join(self.data_dir, 'result.npz'),
                                z_true=z_true, z_pred=z_pred_avg,
                                z_err=z_pred_std_cal, alpha=alpha)
        
        else:
            raise NotImplementedError

    def sigma(self, z_pred, z_spec):
        del_z = z_pred - z_spec
        sigma_nmad = 1.48 * \
            np.median(np.abs((del_z - np.median(del_z))/(1 + z_spec)))
        return np.around(sigma_nmad, 3)


    def eta(self, z_pred, z_spec):
        delt_z = np.abs(z_pred - z_spec) / (1 + z_spec)
        et = np.sum((delt_z > 0.15)) / np.shape(z_pred)[0] * 100
        return np.around(et, 2)


    def plot_result(self, z_true, z_pred, z_err=None, savedir=None):
        
        plt.figure(figsize=(8, 8))
        if z_err is None:
            plt.scatter(z_true, z_pred, c='red', s=3)
        else:
            plt.errorbar(z_true, z_pred, yerr=z_err, fmt='.', c='red',
                         ecolor='lightblue', elinewidth=0.5)
        a = np.arange(7)
        plt.plot(a, 1.15 * a + 0.15, 'k--', alpha=0.5)
        plt.plot(a, 0.85 * a - 0.15, 'k--', alpha=0.5)

        sigma_all = self.sigma(z_pred, z_true)
        eta_all = self.eta(z_pred, z_true)
        zmax = np.max(z_true)
        plt.xlim(0, zmax)
        plt.ylim(0, zmax)
        plt.text(0.5, 3.0, '$\eta={v:.2f}\%$'.format(v=eta_all), fontsize=14)
        plt.text(0.5, 3.2, '$\sigma_{NMAD}=$'+'{:.3f}'.format(sigma_all), fontsize=14)  
        plt.title(f'{self.model_name}', fontsize=16)
        plt.xlabel(r'$z_{\rm true}$')
        plt.ylabel(r'$z_{\rm pred}$')
        plt.savefig(os.path.join(savedir, 'results.png'))

    def create_catalogue(self, catalogue=None, z_pred=None, info_keys=['ra', 'dec'], z_err=None):
        if catalogue is not None:
            with fits.open(catalogue, mode='update') as hdul:
                data = Table(hdul[1].data)

                infos = []
                for key in info_keys:
                    infos.append(data[key])
        else:
            print('Catalogue not provided, only save photo-zs.')

        header = fits.Header()
        header['NUM'] = (z_pred.shape[0], 'Num sources.')
        header['MODEL'] = (self.model_type, 'Deep learning model.')
        header['NAME'] = (self.model_name, 'Model name.')
        header['DATA'] = (self.data_type, 'Data used.')
        
        if self.model_type == 'BNN':
            header['ALPHA'] = (self.alpha, 'Uncertainty calibration parameter.')

        primary_hdu = fits.PrimaryHDU(header=header)
        
        catalog = []
        names = []
        
        if catalogue is not None:
        
            for key, info in zip(info_keys, infos):
                catalog.append(info)
                names.append(key)
        else:
            pass
        
        catalog.append(z_pred)
        names.append('Z_PRED')
        
        if z_err is not None:
            catalog.append(z_err)
            names.append('Z_ERR')
                
        table = Table(catalog, names=names)
        
        table_hdu = fits.BinTableHDU(table)

        hdulist = fits.HDUList([primary_hdu, table_hdu])

        new_cat_name = f'photoz_catalogue.fits'
        catalogue_file = os.path.join(self.data_dir, new_cat_name)
        hdulist.writeto(catalogue_file, overwrite=True)


    def inferece(self, ds, datasize, catalogue=None, info_keys=['ra', 'dec'], n_runs=200):

        if self.model_type == 'NN':
            z_pred = self.model.predict(ds)
            z_pred = z_pred.reshape(-1)

            self.create_catalogue(catalogue, z_pred, info_keys=info_keys)
            
        elif self.model_type == 'BNN':
            
            z_pred_n_runs = np.zeros((n_runs, datasize))

            for i in tqdm(range(n_runs)):
                z_pred_n_runs[i, :] = np.reshape(self.model.predict(ds, verbose=0),
                                                 datasize)
                tf.keras.backend.clear_session()
                gc.collect()

            z_pred_avg = np.mean(z_pred_n_runs, axis=0)
            z_pred_std = np.std(z_pred_n_runs, axis=0)

            z_pred_std_cal = self.alpha * z_pred_std
            
            self.create_catalogue(catalogue, z_pred_avg, info_keys=info_keys, z_err=z_pred_std_cal)

        else:
            raise NotImplementedError