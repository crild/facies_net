# Make initial package imports
import numpy as np
import time

from facies_net_func.segy_files import *
from facies_net_func.training import *
from facies_net_func.visualize import *

### ---- MASTER/MAIN function ----
# Make an overall master function that takes inn some basic parameters,
# trains, predicts, and visualizes the results from a model
def master(segy_filename,cube_incr,train_dict={},pred_dict={},mode = 'full'):
    # segy_filename: filename of the segy-cube to be imported (necessary for copying the segy-frame before writing a new segy)
    # cube_incr: number of increments included in each direction from the example to make a mini-cube
    # train_dict: Training parameters packaged as a Python dictionary
    # pred_dict: Prediciton parameters packaged as a Python dictionary
    # mode: Do we want to train a model('train'), predict using an external model('predict'), or train a model and predict using it('full')

    # Store all the segy-data and specifications as a segy object
    segy_obj = segy_reader(segy_filename)

    # Are we going to perform training?
    if mode == 'train' or mode == 'full':
        # If there is a model given in the prediction dictionary continue training on this model
        if 'keras_model' in pred_dict:
            keras_model = pred_dict['keras_model']
        else:
            keras_model = None

        if 'num_class' in pred_dict:
            num_classes = pred_dict['num_class']
        else:
            keras_model = None

        # Unpack the dictionary of training parameters
        file_list           = train_dict['files']
        num_epochs          = train_dict['epochs']
        num_examples        = train_dict['num_train_ex']
        batch_size          = train_dict['batch_size']
        val_split           = train_dict['val_split']
        opt_patience        = train_dict['opt_patience']
        data_augmentation   = train_dict['data_augmentation']
        write_out           = train_dict['save_model']
        write_location      = train_dict['save_location']

        # Print out an initial statement to confirm the parameters(QC)
        print('num epochs:',num_epochs)
        print('num examples per epoch:',num_examples)
        print('batch size:',batch_size)
        print('optimizer patience:',opt_patience)

        # Time the training process
        start_train_time = time.time()

        # Train a new model/further train the uploaded model and store the result as the model output
        model = train_model(segy_obj = segy_obj,
                            file_list = file_list,
                            cube_incr = cube_incr,
                            num_epochs = num_epochs,
                            num_classes = num_classes,
                            num_examples = num_examples,
                            batch_size = batch_size,
                            val_split = val_split,
                            opt_patience = opt_patience,
                            data_augmentation = data_augmentation,
                            keras_model = keras_model,
                            write_out = write_out,
                            write_location = write_location)

        # Print the time taken for the training
        end_train_time = time.time()
        train_time = end_train_time-start_train_time # seconds

        # print to the user the total time spent training
        if train_time <= 300:
            print('Total time elapsed during training:',train_time, ' sec.')
        elif 300 < train_time <= 60*60:
            minutes = train_time//60
            seconds = (train_time%60)
            print('Total time elapsed during training:',minutes,' min., ',seconds,' sec.')
        elif 60*60 < train_time <= 60*60*24:
            hours = train_time//(60*60)
            minutes = (train_time%(60*60))*(1/60)
            print('Total time elapsed during training:',hours,' hrs., ',minutes,' min., ')
        else:
            days = train_time//(24*60*60)
            hours = (train_time%(24*60*60))*(1/60)*((1/60))
            print('Total time elapsed during training:',days,' days, ',hours,' hrs., ')



    elif mode == 'predict':
        # If we aren't performing any training
        print('Using uploaded model for prediction')
    else:
        print('Invalid mode! Accepted inputs are ''train'', ''predict'', or ''full''')
        return None



    # Are we going to perform prediction?
    if mode == 'predict' or mode == 'full':
        # Let the user know if we have made new computations on the model used for prediction
        if mode == 'full':
            print('Using the newly computed model for prediction')
        else:
            model  = pred_dict['keras_model']

        # Unpack the prediction dictionary
        section_edge  = pred_dict['section_edge']
        xline_ref = pred_dict['xline']
        num_classes = pred_dict['num_class']
        sect_form = pred_dict['cord_syst']
        show_feature  = pred_dict['show_feature']
        save_pred = pred_dict['save_pred']
        save_loc = pred_dict['save_location']
        pred_batch = pred_dict['pred_batch']


        # Time the prediction process
        start_pred_time = time.time()

        # Make a prediction on the master segy object using the desired model, and plot the results
        pred = visualization(filename = segy_filename,
                             seis_obj = segy_obj,
                             keras_model = model,
                             cube_incr = cube_incr,
                             section_edge = section_edge,
                             xline_ref = xline_ref,
                             num_classes = num_classes,
                             sect_form = sect_form,
                             save_pred = save_pred,
                             save_file = save_loc,
                             pred_batch = pred_batch,
                             show_feature = show_feature)

        # Print the time taken for the prediction
        end_pred_time = time.time()
        pred_time = end_pred_time-start_pred_time # seconds

        # print to the user the total time spent training
        if pred_time <= 300:
            print('Total time elapsed during prediction and saving:',pred_time, ' sec.')
        elif 300 < pred_time <= 60*60:
            minutes = pred_time//60
            seconds = (pred_time%60)
            print('Total time elapsed during prediction and saving:',minutes,' min., ',seconds,' sec.')
        elif 60*60 < pred_time <= 60*60*24:
            hours = pred_time//(60*60)
            minutes = (pred_time%(60*60))*(1/60)
            print('Total time elapsed during prediction and saving:',hours,' hrs., ',minutes,' min., ')
        else:
            days = pred_time//(24*60*60)
            hours = (pred_time%(24*60*60))*(1/60)*((1/60))
            print('Total time elapsed during prediction and saving:',days,' days, ',hours,' hrs., ')

    else:
        # Make an empty variable for the prediction output
        pred = None

    # Return the new model and/or prediction as an output dictionary
    output = {
        'model' : model,
        'pred' : pred
    }

    return output
