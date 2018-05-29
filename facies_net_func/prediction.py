# Make initial package imports
import numpy as np
import time
import segyio

from facies_net_func.segy_files import *

from keras.models import Sequential, Model
from shutil import copyfile

### ---- Functions for the prediction part of the program ----
# Parse the cube into sub-cubes suitable as model input
def cube_parse(seis_arr,cube_incr,inline_num,xline_num,depth):
    # seis_arr: a 3D numpy array that holds a seismic cube
    # cube_incr: number of increments included in each direction from the example to make a mini-cube
    # inline_num: what inline do we use?
    # xline_num: what xline do we use?
    # depth: what depth do we use?

    # Make some initial definitions wrt. dimensionality
    cube_size = 2*cube_incr+1
    num_channels = seis_arr.shape[3]
    inline_num -= cube_incr
    xline_num -= cube_incr
    depth -= cube_incr

    # Preallocate the output array, if concatenated it's 5 dimensional, if not it's 7 dimensional
    example = np.empty((1,cube_size,cube_size,cube_size,num_channels),dtype=np.float32)

    # Make the cubes
    example[0,:,:,:,:] = seis_arr[inline_num:inline_num+cube_size,\
                                   xline_num:xline_num+cube_size,\
                                   depth:depth+cube_size,:]

    # Return the list of examples stored as the desired type of array
    return example



# Make an intermediate output model to check filters
def makeIntermediate(keras_model,layer_name):
    # keras_model: keras model that has been trained previously
    # layer_name: name of the layer with the desired output

    # Define the new model that stops at the desired layer
    intermediate_layer_model = Model(inputs=keras_model.input,\
                                     outputs=keras_model.get_layer(layer_name).output)

    # Return the newly defined model
    return intermediate_layer_model



# Predict the output class of the given input traces
def predicting(filename,seis_obj,keras_model,cube_incr,num_classes, section,\
               print_segy = False,savename = 'default_write',\
               pred_batch = 1, show_features = False, layer_name='attribute_layer'):
    # filename: filename of the segy-cube to be imported (necessary for copying the segy-frame before writing a new segy)
    # seis_obj: Object returned from the segy_decomp function
    # keras_model: keras model that has been trained previously
    # cube_incr: number of increments included in each direction from the example to make a mini-cube
    # num_classes: num_classes: number of destinct classes we are training on
    # section: edge locations(index) of the sub-section (min. inline, max. inline, min. xline, max xline, min z, max z)
    # print_segy: whether or not to save the prediction as a segy, npy and csv file (previously just segy)
    # savename: name of the files to be saved (extensions are added automatically)
    # pred_batch: number of traces to predict on at a time
    # show_features: whether or not to get the features or the classes
    # layer_name: optionally give a different layer to get the features from (name defined in keras.model)

    # Define some initial parameters
    num_channels = seis_obj.cube_num
    cube_size = 2*cube_incr+1

    # Preallocate the full prediction array and if the user wants to show the features make the intermediate model,
    if show_features:
        intermediate_layer_model = Model(inputs=keras_model.input,
                                         outputs=keras_model.get_layer(layer_name).output)

        num_classes = intermediate_layer_model.output_shape[-1]

    prediction = np.empty((\
        (section[5]-section[4]+1)*(section[3]-section[2]+1)*(section[1]-section[0]+1),\
                           num_classes),dtype=np.float32)

    # Preallocate the data array to fill for each batch and initiate iterators
    data = np.empty((pred_batch*(section[5]-section[4]+1),cube_size,cube_size,cube_size,num_channels), dtype=np.float32)
    indx = 0
    jndx = 0

    # Calculate how many sets of batches need to be done and define parameters needed for the final batch
    tot_len = (section[1]-section[0]+1)*(section[3]-section[2]+1)
    rem = tot_len % pred_batch
    num_it = tot_len // pred_batch
    # Time the sub_prediction
    start = time.time()

    # Start making sub-cubes from the input traces and store then in the data array
    print('Retrieving to memory:')
    for il_num in range(section[0],section[1]+1):
        # Make a progres update for the inline number
        print('inline-num:',il_num-section[0]+1,'/',section[1]-section[0]+1)
        for xl_num in range(section[2],section[3]+1):
            # Make a progres update for the xline number
            print('xline-num:',xl_num-section[2]+1,'/',section[3]-section[2]+1)
            for z_num in range(section[5]-section[4]+1):
                # Call the cube_parse function to get the cube corresponding to the current point
                data[indx*(section[5]-section[4]+1)+z_num,:,:,:,:] = cube_parse(seis_arr = seis_obj.data,
                                                                                          cube_incr = cube_incr,
                                                                                          inline_num = il_num,
                                                                                          xline_num = xl_num,
                                                                                          depth = z_num+section[4])

            # Check if we have filled up the data array and need to do a prediction
            if (indx+1) % pred_batch == 0:
                print('Making prediction on sub-section:')

                # Predict the given class or features dependant on the user input
                if show_features:
                    prediction[jndx*(pred_batch*(section[5]-section[4]+1)):\
                              (jndx+1)*(pred_batch*(section[5]-section[4]+1)),:] = \
                                    intermediate_layer_model.predict((data))

                else:
                    # Model prediction of classes
                    prediction[jndx*(pred_batch*(section[5]-section[4]+1)):\
                               (jndx+1)*(pred_batch*(section[5]-section[4]+1)),:] = \
                    np.expand_dims(keras_model.predict_classes((data)),axis = 1)

                # Tell the user the section is finished
                print('Section finished!')

                if jndx == 0:
                    # Finish the timer and calculate how long the user should expect the program to take:
                    end = time.time()
                    DT = end-start # seconds per iteration
                    tot_time = num_it*DT+(rem/pred_batch)*DT #seconds

                # Give the user an update regarding the time remaining
                time_rem = (tot_time-DT*(jndx+1))
                if time_rem <= 300:
                    print('Approximate time remaining of the prediction:',time_rem, ' sec.')
                elif 300 < time_rem <= 60*60:
                    minutes = time_rem//60
                    seconds = (time_rem%60)
                    print('Approximate time remaining of the prediction:',minutes,' min., ',seconds,' sec.')
                elif 60*60 < time_rem <= 60*60*24:
                    hours = time_rem//(60*60)
                    minutes = (time_rem%(60*60))*(1/60)
                    print('Approximate time remaining of the prediction:',hours,' hrs., ',minutes,' min., ')
                else:
                    days = time_rem//(24*60*60)
                    hours = (time_rem%(24*60*60))*(1/60)*((1/60))
                    print('Approximate time remaining of the prediction:',days,' days, ',hours,' hrs., ')


                # Update iterators and give updates to user
                indx = 0
                jndx+=1
                print('Retrieving to memory:')

            # Check if we have exhausted the range of data to be predicted and need to finish the function
            elif jndx == num_it and indx == rem-1:
                # Slice the data array to only include the relevant part
                data = data[:indx*(section[5]-section[4]+1)+z_num+1]

                # Make the final prediction
                print('Finalizing prediction:')
                if show_features:
                    prediction[jndx*(pred_batch*(section[5]-section[4]+1)):,:] = \
                                    intermediate_layer_model.predict((data))
                else:
                    prediction[jndx*(pred_batch*(section[5]-section[4]+1)):,:] = \
                                np.expand_dims(keras_model.predict_classes((data)),axis = 1)

            # If we should keep filling the data and not predict yet, simply increase the iterator
            else:
                indx+=1

    # Reshape the prediction to the shape of the desired cube
    print('Reshaping prediction:')
    prediction = prediction.reshape((section[1]-section[0]+1,\
                                     section[3]-section[2]+1,\
                                     section[5]-section[4]+1,num_classes),order='C')
    print('Prediction finished!')

    # Save the prediction as a segy, numpy and csv file
    # NOTE: Everything SEGY is made into 32bit-float to conform to commonly used reading programs
    if print_segy:
        class_row = 0

        print('Saving prediction: ...')

        # Save the numpy file
        np.save(savename + '.npy', prediction)

        # Get the right filename in case the input is given as a list
        if type(filename) is list:
            # Save the segy file using the input filename as a framework
            # Just use the first member of the list as the reference
            input_file = filename[0]
        else:
            # Save the segy file using the input filename as a framework
            input_file=filename

        output_file=savename + '.sgy'

        copyfile(input_file, output_file)

        with segyio.open( output_file, "r+" ) as src:
            # iterate through each inline and update the values
            i = 0
            for ilno in src.ilines:
                src.iline[ilno] = -1*(np.ones((src.iline[ilno].shape),dtype = np.float32))

                if src.ilines[section[0]] <= ilno <= src.ilines[section[1]]:
                    line = src.iline[ilno]
                    line[section[2]:section[3]+1,section[4]:section[5]+1] = prediction[i,:,:,class_row]
                    src.iline[ilno]=line
                    i += 1

        # Print to the user that the function has finished saving
        print('Prediction saved.')

    # Return the prediction array
    return prediction
