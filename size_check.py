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
keras_model = keras.models.load_model('F3/test1.h5')
name = 'conv_layer1'

print(keras_model.get_layer(name).input_shape,keras_model.get_layer(name).output_shape)

name = 'conv_layer2'

print(keras_model.get_layer(name).input_shape,keras_model.get_layer(name).output_shape)

name = 'conv_layer3'

print(keras_model.get_layer(name).input_shape,keras_model.get_layer(name).output_shape)

name = 'conv_layer4'

print(keras_model.get_layer(name).input_shape,keras_model.get_layer(name).output_shape)

#name = 'dense_layer1'

#print(keras_model.get_layer(name).input_shape,keras_model.get_layer(name).output_shape)

name = 'attribute_layer'

print(keras_model.get_layer(name).input_shape,keras_model.get_layer(name).output_shape)

name = 'pre-softmax_layer'

print(keras_model.get_layer(name).input_shape,keras_model.get_layer(name).output_shape)
