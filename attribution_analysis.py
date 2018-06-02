#### ------------- Test for attribution ------------

# Import needed functions and modules
import os
import keras
import matplotlib.pyplot as plt

from facies_net_func.data_cond import *
from facies_net_func.segy_files import *
from facies_net_func.attribution import *
from facies_net_func.feature_vis import *

# Set the RNG
np.random.seed(7)

# Define some parameters
end_str = 'Mirror2T'
keras_model = keras.models.load_model('F3/10_epochs_80000_examples_2T.h5')
cube_incr = 30
segy_filename = ['F3_entire.segy']
file_list =     ['F3_facies.segy']
                #['./class_addresses/Snadd_ilxl.pts',
                # './class_addresses/Other_ilxl.pts'] # list of names of class-adresses

# Store all the segy-data and specifications as a segy object
segy_obj = segy_reader(segy_filename)

print('Making class-adresses')
if int(len(file_list)) <= 1:
    tr_adr,val_adr = convert_segy(segy_name = file_list,
                          save = False,
                          savename = None,
                          ex_adjust = True,
                          val_split = 0.3,
                          mode = 'xline')
else:
    tr_adr,val_adr = convert(file_list = file_list,
                             save = False,
                             savename = None,
                             ex_adjust = True,
                             val_split = 0.3)
print('Finished making class-adresses')

print(tr_adr)

# Define parameters for the generators
tr_params =        {'seis_spec'   : segy_obj,
                    'adr_list'    : tr_adr,
                    'cube_incr'   : cube_incr,
                    'num_classes' : 2, #len(file_list),
                    'batch_size'  : 1,
                    'steps'       : 100,
                    'print_info'  : True}

generator = ex_create(**tr_params)

# image index fram tr_adr (must beless than steps in tr_params)
im_idx = 4 #26/40/41!!/44/48/50 good horizons, 27 fault

test_im, y = generator.data_generation(im_idx)

# Check if we need to make a new image directory
if not os.path.exists('images/image'+str(im_idx)):
    os.makedirs('images/image'+str(im_idx))

save_or(test_im,name = 'images/image'+str(im_idx)+'/Original_im',formatting = 'normalize')

ig = integrated_gradients(keras_model)

save_overlay(ig,len(file_list),test_im,name='images/image'+str(im_idx)+'/overlay'+end_str,steps = 100, mosaic = 'rows')


labl = np.append(y,(np.nonzero(y)[1]*np.ones(y.shape)),axis = 0)
np.savetxt(fname='images/image'+str(im_idx)+'/label',X = labl,fmt='%i')
print('The actual label was:',int(labl[1,0]))
