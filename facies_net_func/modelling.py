### ---- Functions for modelmaking ----

# Make initial package imports
import numpy as np
import keras
from tensorflow.python.keras._impl.keras.engine import InputSpec
from tensorflow.python.keras._impl.keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Conv3D, Dropout
from keras.layers.normalization import BatchNormalization

# Define 3D dropout taken from TensorFlow Documentation
class SpatialDropout3D(Dropout):
  """Spatial 3D version of Dropout.
  This version performs the same function as Dropout, however it drops
  entire 3D feature maps instead of individual elements. If adjacent voxels
  within feature maps are strongly correlated (as is normally the case in
  early convolution layers) then regular dropout will not regularize the
  activations and will otherwise just result in an effective learning rate
  decrease. In this case, SpatialDropout3D will help promote independence
  between feature maps and should be used instead.
  Arguments:
      rate: float between 0 and 1. Fraction of the input units to drop.
      data_format: 'channels_first' or 'channels_last'.
          In 'channels_first' mode, the channels dimension (the depth)
          is at index 1, in 'channels_last' mode is it at index 4.
          It defaults to the `image_data_format` value found in your
          Keras config file at `~/.keras/keras.json`.
          If you never set it, then it will be "channels_last".
  Input shape:
      5D tensor with shape:
      `(samples, channels, dim1, dim2, dim3)` if data_format='channels_first'
      or 5D tensor with shape:
      `(samples, dim1, dim2, dim3, channels)` if data_format='channels_last'.
  Output shape:
      Same as input
  References:
      - [Efficient Object Localization Using Convolutional
        Networks](https://arxiv.org/abs/1411.4280)
  """

  def __init__(self, rate, data_format=None, **kwargs):
    super(SpatialDropout3D, self).__init__(rate, **kwargs)
    if data_format is None:
      data_format = K.image_data_format()
    if data_format not in {'channels_last', 'channels_first'}:
      raise ValueError('data_format must be in '
                       '{"channels_last", "channels_first"}')
    self.data_format = data_format
    self.input_spec = InputSpec(ndim=5)

  def _get_noise_shape(self, inputs):
    input_shape = K.shape(inputs)
    if self.data_format == 'channels_first':
      return (input_shape[0], input_shape[1], 1, 1, 1)
    elif self.data_format == 'channels_last':
      return (input_shape[0], 1, 1, 1, input_shape[4])


### ---- Make the model for the neural network ----
def make_model(cube_size = 65, num_channels = 1, num_classes = 2,\
               opt = keras.optimizers.adam(lr=0.001)):
    #  This model is loosely built after that of Anders Waldeland (5 Convolutional layers
    #  and 2 fully connected layers with rectified linear and softmax activations)
    #  We have added drop out and batch normalization our selves, and experimented with multi-prediction
    #
    #  We also use the Adam optimizer with a given learning rate (Note that this is adapted later)
    #
    model = Sequential()
    model.add(Conv3D(50, (5, 5, 5), padding='valid', input_shape=(cube_size,cube_size,cube_size,num_channels), strides=(4, 4, 4), \
                     data_format="channels_last",name = 'conv_layer1'))
    model.add(BatchNormalization())
    model.add(SpatialDropout3D(0.2))
    model.add(Activation('relu'))

    model.add(Conv3D(50, (3, 3, 3), strides=(2, 2, 2), padding = 'valid',name = 'conv_layer2'))
    model.add(BatchNormalization())
    model.add(SpatialDropout3D(0.2))
    model.add(Activation('relu'))

    model.add(Conv3D(50, (3, 3, 3), strides=(2, 2, 2), padding= 'valid',name = 'conv_layer3'))
    model.add(BatchNormalization())
    model.add(SpatialDropout3D(0.2))
    model.add(Activation('relu'))

    model.add(Conv3D(50, (3, 3, 3), strides=(1, 1, 1), padding= 'valid',name = 'conv_layer4'))
    model.add(BatchNormalization())
    model.add(SpatialDropout3D(0.2))
    model.add(Activation('relu'))

    model.add(Dense(50,name = 'dense_layer1'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Activation('relu'))

    model.add(Dense(10,name = 'attribute_layer'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Activation('relu'))

    model.add(Dense(num_classes, name = 'pre-softmax_layer'))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))
    model.add(Flatten())


    # Compile the model with the desired loss, optimizer, and metric
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    return model
