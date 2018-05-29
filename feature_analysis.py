### --------- function for making feature images from non-random starts --------

# Import needed functions and modules
import os
import keras

from facies_net_func.data_cond import *
from facies_net_func.segy_files import *
from facies_net_func.feature_vis import *

# Set the RNG
np.random.seed(7)

# Define some parameters
end_str = 'baseline'
keras_model = keras.models.load_model('Hoop/25_epochs_50000_examples2_'+ end_str +'.h5')
layer_name = 'pre-softmax_layer' #'conv_layer2' #'attribute_layer'
cube_incr = 30
segy_filename = ['Hoop_crop2.sgy']
file_list =     ['Facies2.sgy']
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
                                  val_split = 0.3)
else:
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
                    'num_classes' : 5, #len(file_list),
                    'batch_size'  : 1,
                    'steps'       : 100,
                    'print_info'  : True}

generator = ex_create(**tr_params)

# image index fram tr_adr (must beless than steps in tr_params)
im_idx = 1 #26/40/41!!/44/48/50 good horizons, 27 fault

start_im, y = generator.data_generation(im_idx)

# Check if we need to make a new image directory
if not os.path.exists('images/image'+str(im_idx)):
    os.makedirs('images/image'+str(im_idx))

save_or(start_im,name = 'images/image'+str(im_idx)+'/Original_im',formatting = 'normalize')

(filter_list, losses) = features(keras_model,
                                 layer_name,
                                 iterations = 100,
                                 smoothing_par = None,
                                 inp_im = start_im,
                                 name = 'images/image'+str(im_idx)+'/pre-softmax'+end_str)

(filter_list, losses) = features(keras_model,
                                 layer_name,
                                 iterations = 100,
                                 smoothing_par = None,
                                 inp_im = None,
                                 name = 'images/image'+str(im_idx)+'/pre-softmax2'+end_str)

layer_name = 'conv_layer2'

(filter_list, losses) = features(keras_model,
                                 layer_name,
                                 iterations = 100,
                                 smoothing_par = None,
                                 inp_im = start_im,
                                 name = 'images/image'+str(im_idx)+'/conv_layer2'+end_str)

(filter_list, losses) = features(keras_model,
                                 layer_name,
                                 iterations = 100,
                                 smoothing_par = None,
                                 inp_im = None,
                                 name = 'images/image'+str(im_idx)+'/conv_layer2'+end_str)


np.set_printoptions(precision=2)
print('Loss list:')
print(losses)
