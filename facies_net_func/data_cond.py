# Make initial package imports
import numpy as np
import random
import time
import keras
import segyio

### ---- Functions for data conditioning part of the program ----
# Make a function that combines the adress cubes and makes a list of class adresses
def convert(file_list, save = False, savename = 'adress_list', ex_adjust = False, val_split = 0):
    # file_list: list of file names(strings) of adresses for the different classes
    # save: boolean that determines if a new ixz file should be saved with adresses and class numbers
    # savename: desired name of new .ixz-file
    # ex_adjust: boolean that determines if the amount of each class should be approximately equalized

    # preallocate lists and arrays
    len_array = np.zeros(len(file_list),dtype = np.int32)
    tr_len = np.zeros(len(file_list),dtype = np.int32)
    val_len = np.zeros(len(file_list),dtype = np.int32)
    tr_list = np.empty([0,4], dtype = np.int32)
    val_list = np.empty([0,4], dtype = np.int32)

    # Itterate through the list of example adresses and store the class as an integer
    for i in range(len(file_list)):
        # Upload file number i, store its length and shuffle it
        a = np.loadtxt(file_list[i], skiprows=0, usecols = range(3), dtype = np.int32)
        len_array[i] = len(a)
        np.random.shuffle(a)

        # Append the examples to the right list
        tr_len[i] = int((1-val_split)*len_array[i])
        val_len[i] = len_array[i] - tr_len[i]
        tr_list = np.append(tr_list,np.append(a[:tr_len[i]],i*np.ones((tr_len[i],1),dtype = np.int32),axis=1),axis=0)
        val_list = np.append(val_list,np.append(a[tr_len[i]:],i*np.ones((val_len[i],1),dtype = np.int32),axis=1),axis=0)

    # If desired multiply up the classes needed to balance out the training data
    if ex_adjust:
        # Calculate the multipliers
        multiplier = len_array/max(len_array)
        multiplier = 1//multiplier

        i = 0
        for mult in multiplier-1:
            if mult != 0:
                # load data from the right file and make the multiplier into an int
                a = np.loadtxt(file_list[i], skiprows=0, usecols = range(3), dtype = np.int32)

                # append more data to the right file
                tr_list = np.append(tr_list,np.tile(np.append(a[:tr_len[i]],i*np.ones((tr_len[i],1),dtype = np.int32),axis=1),(int(mult), 1)),axis=0)
                val_list = np.append(val_list,np.tile(np.append(a[tr_len[i]:],i*np.ones((val_len[i],1),dtype = np.int32),axis=1),(int(mult), 1)),axis=0)

            # increase the iterator
            i += 1

    # shuffle the datasets:
    np.random.shuffle(tr_list)
    np.random.shuffle(val_list)

    # Add the option to save it as an external file
    if save:
        # save the file as the given str-name
        np.savetxt(savename + 'tr.ixz', tr_list, fmt = '%i')

        # if there are validation data save these too
        if val_split != 0:
            np.savetxt(savename + 'val.ixz', val_list, fmt = '%i')

    # Return the list of adresses and classes as a numpy array
    if val_split != 0:
        return tr_list, val_list
    else:
        return tr_list


### ---- Functions for data conditioning part of the program ----
# Make a function that combines the adress cubes and makes a list of class adresses
def convert_segy(segy_name, save = False, savename = 'adress_list', ex_adjust = False, val_split = 0, mode = 'iline'):
    # segy_name: name of the segy-cube with input data
    # save: boolean that determines if a new ixz file should be saved with adresses and class numbers
    # savename: desired name of new .ixz-file
    # ex_adjust: boolean that determines if the amount of each class should be approximately equalized

    # Make an empty object to hold the output data
    output = segyio.spec()

    # open the segyfile and start decomposing it
    with segyio.open(segy_name[0], "r" ) as segyfile:
        # Memory map file for faster reading (especially if file is big...)
        segyfile.mmap()

        if mode == 'xline':
            # get out a x-line
            xl_index = int(np.floor(segyfile.iline.len/2))
            xline_num = segyfile.xlines[xl_index]
            print('Training on xline:',xline_num)
            data = segyfile.xline[xline_num]

            # Get some initial parameters of the data
            (ilen,zlen) = data.shape
            ils = segyfile.ilines

            zs = segyfile.samples

            # Preallocate the array that we want to make
            full_np = np.empty((ilen*zlen,4),dtype = np.int32)
            i = 0

            # Itterate through the numpy-cube and convert each trace individually to a section of csv
            for il in ils:
                # Make a list of the inline number, xline number, and depth for the given trace
                I = il*(np.ones((zlen,1)))
                X = xline_num*(np.ones((zlen,1)))
                Z = np.expand_dims(zs,axis=1)

                # Store the predicted class/probability at each of the given depths of the trace
                D = np.expand_dims(data[i,:],axis=1)

                # Concatenate these lists together and insert them into the full array
                inp_li = np.concatenate((I,X,Z,D),axis=1)
                full_np[i*zlen:(i+1)*zlen,:] = inp_li
                i+=1

            split_idx = int(np.floor((1-val_split)*ilen*zlen))

        elif mode == 'iline':
            # get out a in-line
            il_index = int(np.floor(segyfile.xline.len/2))
            iline_num = segyfile.ilines[il_index]
            print('Training on inline:',iline_num)
            data = segyfile.iline[iline_num]

            # Get some initial parameters of the data
            (xlen,zlen) = data.shape
            xls = segyfile.xlines

            zs = segyfile.samples

            # Preallocate the array that we want to make
            full_np = np.empty((xlen*zlen,4),dtype = np.int32)
            i = 0

            # Itterate through the numpy-cube and convert each trace individually to a section of csv
            for xl in xls:
                # Make a list of the inline number, xline number, and depth for the given trace
                I = iline_num*(np.ones((zlen,1)))
                X = xl*(np.ones((zlen,1)))
                Z = np.expand_dims(zs,axis=1)

                # Store the predicted class/probability at each of the given depths of the trace
                D = np.expand_dims(data[i,:],axis=1)

                # Concatenate these lists together and insert them into the full array
                inp_li = np.concatenate((I,X,Z,D),axis=1)
                full_np[i*zlen:(i+1)*zlen,:] = inp_li
                i+=1

            split_idx = int(np.floor((1-val_split)*xlen*zlen))

        else:
            print('Unknown mode!!!')
            return



    np.random.shuffle(full_np)

    tr_list = full_np[:split_idx]
    val_list = full_np[split_idx:]

    # Add the option to save it as an external file
    if save:
        # save the file as the given str-name
        np.savetxt(savename + 'tr.ixz', tr_list, fmt = '%i')

        # if there are validation data save these too
        if val_split != 0:
            np.savetxt(savename + 'val.ixz', val_list, fmt = '%i')

    # Return the list of adresses and classes as a numpy array
    if val_split != 0:
        return tr_list, val_list
    else:
        return tr_list

# Function for example creating generators
# Outputs a dictionary with pairs of cube tuples and labels
class ex_create(keras.utils.Sequence):
    # keras.utils.Sequence is a standard keras generator type

    # Initiation of the class, run once
    def __init__(self, seis_spec, adr_list, cube_incr, num_classes, batch_size, steps, data_augmentation=['False'], print_info = False):
        # self: stadard class notation
        # seis_spec: object that holds the specifications of the seismic cube;
        # adr_list: 4-column numpy matrix that holds a header in the first row, then adress and class information for examples
        # cube_incr: the number of increments included in each direction from the example to make a mini-cube
        # num_classes: number of classes they data can be divided into
        # batch_size: the size of each batch of data that is created
        # steps: the maximum number of batches that should be created
        # print_info: whether or not we should print the segy-boundary information

        # Define the cube size and number of surveys
        self.cube_incr = cube_incr
        self.cube_size = 2*cube_incr+1
        self.num_channels = seis_spec.cube_num
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.steps = steps
        self.adr_list = adr_list
        self.n = 0
        self.seis_arr = seis_spec.data
        self.data_augmentation = data_augmentation

        # Define some boundary parameters given in the input object
        self.inline_start = seis_spec.inl_start
        self.inline_end = seis_spec.inl_end
        self.inline_step = seis_spec.inl_step
        self.xline_start = seis_spec.xl_start
        self.xline_end = seis_spec.xl_end
        self.xline_step = seis_spec.xl_step
        self.t_start = seis_spec.t_start
        self.t_end = seis_spec.t_end
        self.t_step = seis_spec.t_step

        # Define the buffer zone around the edge of the cube that defines the legal/illegal adresses
        self.inl_min = self.inline_start + self.inline_step*self.cube_incr
        self.inl_max = self.inline_end - self.inline_step*self.cube_incr
        self.xl_min = self.xline_start + self.xline_step*self.cube_incr
        self.xl_max = self.xline_end - self.xline_step*self.cube_incr
        self.t_min = self.t_start + self.t_step*self.cube_incr
        self.t_max = self.t_end - self.t_step*self.cube_incr

        if print_info:
            # Print the buffer zone edges
            print('Defining the buffer zone:')
            print('(inl_min,','inl_max,','xl_min,','xl_max,','t_min,','t_max)')
            print('(',self.inl_min,',',self.inl_max,',',self.xl_min,',',self.xl_max,',',self.t_min,',',self.t_max,')')

            # Also give the buffer values in terms of indexes
            print('(',self.cube_incr,',',((self.inline_end-self.inline_start)//self.inline_step) - self.cube_incr,\
                  ',',self.cube_incr,',',((self.xline_end-self.xline_start)//self.xline_step) - self.cube_incr,\
                  ',',self.cube_incr,',',((self.t_end-self.t_start)//self.t_step) - self.cube_incr,')')


    def __len__(self):
        # define the number of batches in an epoch
        return int(self.steps)


    # Start the generators
    def __getitem__(self, index):
        # self: standard class notation
        # index: batch number (starts at 0)

        # define the starting point in the next array:
        idx_start = index*self.batch_size

        # make the next data batch
        X, y = self.data_generation(idx_start)

        return X, y


    # Make a batch of data (potentially implement augmentation)
    def data_generation(self,index_start):
        # index_start: starting point of array

        # preallocate the examples and labels arrays
        examples = np.empty((self.batch_size,self.cube_size,self.cube_size,self.cube_size,self.num_channels),dtype=np.float32)
        labels = np.empty((self.batch_size),dtype=np.int8)

        j = 0
        while j < self.batch_size:
            # test if the current adress is "legal"
            ji = index_start + j
            adr = self.adr_list[ji]

            if (adr[0]>=self.inl_min and adr[0]<self.inl_max) and \
                (adr[1]>=self.xl_min and adr[1]<self.xl_max) and \
                (adr[2]>=self.t_min and adr[2]<self.t_max):

                # Make the example for the given address
                # Convert the adresses to indexes and store the examples in the 4th dimension
                idx = [(adr[0]-self.inline_start)//self.inline_step,
                       (adr[1]-self.xline_start)//self.xline_step,
                       (adr[2]-self.t_start)//self.t_step]

                examples[j,:,:,:,:] = self.seis_arr[idx[0]-self.cube_incr:idx[0]+self.cube_incr+1,\
                              idx[1]-self.cube_incr:idx[1]+self.cube_incr+1,\
                              idx[2]-self.cube_incr:idx[2]+self.cube_incr+1,:]

                # Iterate through the data augmentations
                for operat in self.data_augmentation:
                    # Do the data augmentation,
                    # Probability of augmentation is here 50% (.5)
                    if operat == 'Mirror1':
                        if np.random.uniform(0,1,1) > .5:
                            # Mirror in width
                            examples[j,:,:,:,:] = examples[j,::-1,:,:,:]

                    elif operat == 'Mirror2':
                        if np.random.uniform(0,1,1) > .5:
                            # Mirror in length
                            examples[j,:,:,:,:] = examples[j,:,::-1,:,:]

                    elif operat == 'Mirror3':
                        if np.random.uniform(0,1,1) > .5:
                            # Mirror in length
                            examples[j,:,:,:,:] = examples[j,:,:,::-1,:]

                    elif operat == 'Transpose':
                        if np.random.uniform(0,1,1) > .5:
                            # Transpose
                            examples[j,:,:,:,:] = np.transpose(examples[j,:,:,:,:],(1,0,2,3))

                    elif operat == 'Mirror1T':
                        if np.random.uniform(0,1,1) > .5:
                            # Mirror in width
                            examples[j,:,:,:,:] = examples[j,::-1,:,:,:]
                            examples[j,:,:,:,:] = np.transpose(examples[j,:,:,:,:],(1,0,2,3))

                    elif operat == 'Mirror2T':
                        if np.random.uniform(0,1,1) > .5:
                            # Mirror in length
                            examples[j,:,:,:,:] = examples[j,:,::-1,:,:]
                            examples[j,:,:,:,:] = np.transpose(examples[j,:,:,:,:],(1,0,2,3))

                    elif operat == 'Mirror12T':
                        if np.random.uniform(0,1,1) > .5:
                            # Mirror in length
                            examples[j,:,:,:,:] = examples[j,::-1,:,:,:]
                            examples[j,:,:,:,:] = examples[j,:,::-1,:,:]
                            examples[j,:,:,:,:] = np.transpose(examples[j,:,:,:,:],(1,0,2,3))

                # Put the label into the lists
                labels[j] = adr[-1]

                # increase the iterator for number of values filled into lists
                j+=1

            else:
                # remove the illegal entry from the current file
                self.adr_list = np.delete(self.adr_list, (ji), axis = 0)
                # increase the iterator for illegal examples
                self.n+=1
                if ji // self.n < 10 and self.n % 1000 == 0:
                    print('\nWarning! Poorly conditioned dataset!')
                    print('So far there have been '+str(self.n)+' illegal examples out of '+str(ji)+'('+str(int((self.n*100)/ji))+'%).')

        if ji+1 == self.steps*self.batch_size:
            self.n = 0

        return examples, keras.utils.to_categorical(labels, self.num_classes)


    def on_epoch_end(self):
        # define data augmentation at the end of an epoch
        return
