from dataProcess import DataProcess
from photzEstimation import PhotzEstimation
import numpy as np

images = np.random.normal((100, 32, 32, 7))
flux = np.random.normal((100, 7))
err_flux = np.random.normal((100, 7))

dataprocess = DataProcess('image', mode='evaluate', batch_size=32)
dataprocess.load_images(images)

ds, ds_z = dataprocess.get_dataset()

# dataprocess = DataProcess('image', mode='train', batch_size=1024)
# dataprocess.load_catalogue('catalogue_filename')
# dataprocess.load_images(imgnames='imgnames')

# dataprocess.load_test_catalogue('test_catalogue_filename')
# dataprocess.load_test_images('image')

# ds, datasize = dataprocess.get_dataset()
# test_ds, test_ds_specz = dataprocess.get_test_dataset()

# estimator = PhotzEstimation(model_type='CNN', data_type='image')
# estimator.get_model(datasize)
# estimator.train(ds)
# estimator.evaluate(test_ds, test_ds_specz)