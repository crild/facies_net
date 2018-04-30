# Make initial package imports
import numpy as np
import keras
import time

from facies_net_func.data_cond import *
from facies_net_func.modelling import *

from keras.callbacks import EarlyStopping, LearningRateScheduler, TensorBoard

### ---- Functions for the training part of the program ----
# Function that takes the epoch as input and returns the desired learning rate
def adaptive_lr(input_int):
    # input_int: the epoch that is currently being entered

    # define the learning rate (quite arbitrarily decaying)
    lr = 0.1**input_int

    #return the learning rate
    return lr


# Make the network structure and outline, and train it
def train_model(segy_obj,file_list,cube_incr,num_epochs = 20,\
                num_examples = 50000,batch_size = 32,val_split = 0.2,\
                opt_patience = 5,data_augmentation = False,keras_model = None,\
                write_out = False,write_location = 'default_write'):
    # segy_obj: Object returned from the segy_decomp function
    # file_list: numpy array of class adresses and type, returned from the convert function
    # cube_incr: number of increments included in each direction from the example to make a mini-cube
    # num_epochs: number of epochs we train on a given ensemble of training data
    # num_examples: number of examples we want to train on
    # batch_size: number of mini-batches we go through at a time from the number of examples
    # val_split: what fraction of the data should be validation (the rest will be training)
    # opt_patience: epochs that can pass without improvement in accuracy before the system breaks the loop
    # data_augmentation: boolean which determines whether or not to apply augmentation on the examples
    # keras_model: existing keras model to be improved if the user wants to improve and not create a new model
    # write_out: boolean; save the trained model to disk or not,
    # write_location: desired location on the disk for the model to be saved

    # Calculate som initial parameters
    cube_size = 2*cube_incr+1
    num_classes = len(file_list)
    num_channels = segy_obj.cube_num
    tr_steps = (num_examples*(1-val_split))//batch_size
    val_steps = (num_examples//batch_size) - tr_steps

    # Check if the user wants to make a new model, or train an existing input model
    if keras_model == None:
        # Make a model by passing the size parameters to a model function
        model = make_model(cube_size = cube_size,
                           num_channels = num_channels,
                           num_classes = num_classes)


    else:
        # Define the model we are performing training on as the input model
        model = keras_model

    # Create arrays holding the adresses of training and validation data
    print('Making class-adresses')
    tr_adr,val_adr = convert(file_list = file_list,
                             save = False,
                             savename = None,
                             ex_adjust = True,
                             val_split = val_split)
    print('Finished making class-adresses')

    # Warn the user if they might be about to exhaust their dataset
    if num_examples > 0.8*(len(tr_adr)+len(val_adr)):
        print('\nWarning! The total number of examples are ', str(len(tr_adr)+len(val_adr))+'.')
        print('Training on more than 90% of this can cause errors due to illegal examples!\n')

    # Define parameters for the generators
    tr_params =        {'seis_spec'   : segy_obj,
                        'adr_list'    : tr_adr,
                        'cube_incr'   : cube_incr,
                        'num_classes' : num_classes,
                        'batch_size'  : batch_size,
                        'steps'       : tr_steps,
                        'print_info'  : True}

    val_params =        {'seis_spec'  : segy_obj,
                        'adr_list'    : val_adr,
                        'cube_incr'   : cube_incr,
                        'num_classes' : num_classes,
                        'batch_size'  : batch_size,
                        'steps'       : val_steps}


    # Initiate the example generators
    tr_generator = ex_create(**tr_params)
    val_generator = ex_create(**val_params)

    # Define some initial parameters, and the early stopping and adaptive learning rate callback
    early_stopping  = EarlyStopping(monitor='acc', patience=opt_patience)
    LR_sched        = LearningRateScheduler(schedule = adaptive_lr)
    tensor_board    = TensorBoard(log_dir='./logs/'+write_location,#histogram_freq=1,
                                  write_graph=True, write_images=True)
                                  #batch_size=32,write_grads=True,\
                                  #embeddings_freq=1, embeddings_layer_names=None,
                                  #embeddings_metadata=None)


    # Run the model training
    history = model.fit_generator(generator = tr_generator,
                                  validation_data = val_generator,
                                  callbacks=[early_stopping, tensor_board],
                                  epochs=num_epochs,
                                  shuffle=False)

    # Print the training summary
    print(model.summary())


    # Save the trained model if the user has chosen to do so
    if write_out:
        print('Saving model: ...')
        model.save(write_location + '.h5')
        print('Model saved.')


    # Return the trained model
    return model
