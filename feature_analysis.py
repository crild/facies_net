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
                    'num_classes' : 2, #len(file_list),
                    'batch_size'  : 1,
                    'steps'       : 100,
                    'print_info'  : True}

generator = ex_create(**tr_params)

# image index fram tr_adr (must beless than steps in tr_params)
im_idx = 2 #26/40/41!!/44/48/50 good horizons, 27 fault

start_im, y = generator.data_generation(im_idx)

# Check if we need to make a new image directory
if not os.path.exists('images/image'+str(im_idx)):
    os.makedirs('images/image'+str(im_idx))

save_or(start_im,name = 'images/image'+str(im_idx)+'/Original_im',formatting = 'normalize')

layer_name = 'conv_layer1'

(filter_list, losses) = features(keras_model,
                                 layer_name,
                                 iterations = 100,
                                 smoothing_par = None,
                                 inp_im = start_im,
                                 name = 'images/image'+str(im_idx)+'/conv_layer1_im'+end_str)

(filter_list, losses) = features(keras_model,
                                 layer_name,
                                 iterations = 100,
                                 smoothing_par = None,
                                 inp_im = None,
                                 name = 'images/image'+str(im_idx)+'/conv_layer1_gray'+end_str)

layer_name = 'conv_layer2'

(filter_list, losses) = features(keras_model,
                                 layer_name,
                                 iterations = 100,
                                 smoothing_par = None,
                                 inp_im = start_im,
                                 name = 'images/image'+str(im_idx)+'/conv_layer2_im'+end_str)

(filter_list, losses) = features(keras_model,
                                 layer_name,
                                 iterations = 100,
                                 smoothing_par = None,
                                 inp_im = None,
                                 name = 'images/image'+str(im_idx)+'/conv_layer2_gray'+end_str)

layer_name = 'conv_layer3'

(filter_list, losses) = features(keras_model,
                                 layer_name,
                                 iterations = 100,
                                 smoothing_par = None,
                                 inp_im = start_im,
                                 name = 'images/image'+str(im_idx)+'/conv_layer3_im'+end_str)

(filter_list, losses) = features(keras_model,
                                 layer_name,
                                 iterations = 100,
                                 smoothing_par = None,
                                 inp_im = None,
                                 name = 'images/image'+str(im_idx)+'/conv_layer3_gray'+end_str)

layer_name = 'conv_layer4'

(filter_list, losses) = features(keras_model,
                                 layer_name,
                                 iterations = 100,
                                 smoothing_par = None,
                                 inp_im = start_im,
                                 name = 'images/image'+str(im_idx)+'/conv_layer4_im'+end_str)

(filter_list, losses) = features(keras_model,
                                 layer_name,
                                 iterations = 100,
                                 smoothing_par = None,
                                 inp_im = None,
                                 name = 'images/image'+str(im_idx)+'/conv_layer4_gray'+end_str)


layer_name = 'attribute_layer'

(filter_list, losses) = features(keras_model,
                                 layer_name,
                                 iterations = 100,
                                 smoothing_par = None,
                                 inp_im = start_im,
                                 name = 'images/image'+str(im_idx)+'/attribute_layer_im'+end_str)

(filter_list, losses) = features(keras_model,
                                 layer_name,
                                 iterations = 100,
                                 smoothing_par = None,
                                 inp_im = None,
                                 name = 'images/image'+str(im_idx)+'/attribute_layer_gray'+end_str)


layer_name = 'pre-softmax_layer' #'attribute_layer'

(filter_list, losses) = features(keras_model,
                                 layer_name,
                                 iterations = 100,
                                 smoothing_par = None,
                                 inp_im = start_im,
                                 name = 'images/image'+str(im_idx)+'/pre-softmax_im'+end_str)

(filter_list, losses) = features(keras_model,
                                 layer_name,
                                 iterations = 100,
                                 smoothing_par = None,
                                 inp_im = None,
                                 name = 'images/image'+str(im_idx)+'/pre-softmax_gray'+end_str)


np.set_printoptions(precision=2)
print('Loss list:')
print(losses)
