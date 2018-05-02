#### ------------- Test for attribution ------------

# Import needed functions and modules
import keras
import matplotlib.pyplot as plt

from facies_net_func.data_cond import *
from facies_net_func.segy_files import *
from facies_net_func.attribution import *
from facies_net_func.feature_vis import *


# Define some parameters
keras_model = keras.models.load_model('F3/test2.h5')
cube_incr = 30
segy_filename = ['F3_entire.segy']
file_list = ['./class_addresses/multi_else_ilxl.pts','./class_addresses/multi_grizzly_ilxl.pts',
             './class_addresses/multi_high_amp_continuous_ilxl.pts','./class_addresses/multi_high_amplitude_ilxl.pts',
             './class_addresses/multi_low_amp_dips_ilxl.pts','./class_addresses/multi_low_amplitude_ilxl.pts',
             './class_addresses/multi_low_coherency_ilxl.pts','./class_addresses/multi_salt_ilxl.pts',
             './class_addresses/multi_steep_dips_ilxl.pts'] # list of names of class-adresses

# Store all the segy-data and specifications as a segy object
segy_obj = segy_reader(segy_filename)

print('Making class-adresses')
tr_adr,val_adr = convert(file_list = file_list,
                         save = False,
                         savename = None,
                         ex_adjust = True,
                         val_split = 0.3)
print('Finished making class-adresses')

# Define parameters for the generators
tr_params =        {'seis_spec'   : segy_obj,
                    'adr_list'    : tr_adr,
                    'cube_incr'   : cube_incr,
                    'num_classes' : len(file_list),
                    'batch_size'  : 1,
                    'steps'       : 100,
                    'print_info'  : True}

generator = ex_create(**tr_params)

# image index fram tr_adr (must beless than steps in tr_params)
im_idx = 50

test_im, y = generator.data_generation(im_idx)

save_or(test_im,name = 'images/Original_im',formatting = 'normalize')

ig = integrated_gradients(keras_model)

for i in range(9):
    explanation = np.expand_dims(ig.explain(test_im[0],outc=i,num_steps=200,verbose=1),axis=0)
    save_or(explanation,name = 'images/explanation'+str(i),formatting = 'normalize')
