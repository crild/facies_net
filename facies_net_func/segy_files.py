# Make initial package imports
import segyio
import numpy as np


### ---- Functions for Input data(SEG-Y) formatting and reading ----
# Make a function that takes in a list of segy-filenames and return a speciications object + data
def segy_reader(segy_filenames):
    # segy_filenames: filename of the segy-cubes to be imported

    # Implement more than one segy-cube if the input segy_filenames is a list
    if type(segy_filenames) is str or (type(segy_filenames) is list and len(segy_filenames) == 1):
        # Check if the filename needs to be retrieved from a list
        if type(segy_filenames) is list:
            segy_filenames = segy_filenames[0]

        # Make a master segy object
        segy_obj = segy_decomp(segy_file = segy_filenames)

        # Define how many segy-cubes we're dealing with
        segy_obj.cube_num = 1
        segy_obj.data = np.expand_dims(segy_obj.data, axis = 4)

    elif type(segy_filenames) is list:
        # start an iterator
        i = 0

        # iterate through the list of cube names and store them in a masterobject
        for filename in segy_filenames:
            # Make a master segy object
            if i == 0:
                print('Starting SEG-Y decompressor')
                segy_obj = segy_decomp(segy_file = filename)
                print('Finished using the SEG-Y decompressor')

                # Define how many segy-cubes we're dealing with
                segy_obj.cube_num = len(segy_filenames)

                # Reshape and preallocate the numpy-array for the rest of the cubes
                print('Starting restructuring to 4D arrays')
                ovr_data = np.empty((list(segy_obj.data.shape) + [len(segy_filenames)]))
                ovr_data[:,:,:,i] = segy_obj.data
                segy_obj.data = ovr_data
                ovr_data = None
                print('Finished restructuring to 4D arrays')

            else:
                # Add another cube to the numpy-array
                print('Adding another segy-layer to the 4D array')
                segy_obj.data[:,:,:,i] = segy_adder(segy_file = filename)
                print('Finished adding the segy-layer')

            # Increase the itterator
            i+=1
    else:
        print('The input filename needs to be a string, or a list of strings')

    # return the segy object
    return segy_obj


# Make a function that decompresses a segy-cube and creates a numpy array, and
# a dictionary with the specifications, like in-line range and time step length, etc.
def segy_decomp(segy_file):
    # segy_file: filename of the segy-cube to be imported

    # Make an empty object to hold the output data
    output = segyio.spec()

    # open the segyfile and start decomposing it
    with segyio.open(segy_file, "r" ) as segyfile:
        # Memory map file for faster reading (especially if file is big...)
        segyfile.mmap()

        # Store some initial object attributes
        output.inl_start = segyfile.ilines[0]
        output.inl_end = segyfile.ilines[-1]
        output.inl_step = segyfile.ilines[1] - segyfile.ilines[0]

        output.xl_start = segyfile.xlines[0]
        output.xl_end = segyfile.xlines[-1]
        output.xl_step = segyfile.xlines[1] - segyfile.xlines[0]

        output.t_start = int(segyfile.samples[0])
        output.t_end = int(segyfile.samples[-1])
        output.t_step = int(segyfile.samples[1] - segyfile.samples[0])


        ## NOTE: 'full' for some reason invokes float32 data
        # Pre-allocate a numpy array that holds the SEGY-cube
        output.data = np.empty((segyfile.xline.len,segyfile.iline.len,\
                        (output.t_end - output.t_start)//output.t_step+1), dtype = np.float32)

        output.data = segyio.tools.cube(segy_file)

        # Convert the numpy array to span between -127 and 127
        factor = 127/np.amax(np.absolute(output.data))
        output.data *= factor

    # Return the output object
    return output


# Make a function that adds another layer to a segy-cube
def segy_adder(segy_file, inp_cube):
    # segy_file: filename of the segy-cube to be imported
    # inp_cube: the existing cube that we should add a layer to

    # Make a variable to hold the shape of the input cube and preallocate a data holder
    cube_shape = inp_cube.shape
    output = np.empty(cube_shape[0:-1])

    # open the segyfile and start decomposing it
    with segyio.open(segy_file, "r" ) as segyfile:
        # Memory map file for faster reading (especially if file is big...)
        segyfile.mmap()

        ## NOTE: 'full' for some reason invokes float32 data
        output[:,:,:] = segyio.tools.cube(segy_file)

        # Convert the numpy array to span between -127 and 127 and convert to the desired format
        factor = 127/np.amax(np.absolute(output))
        output *= factor

    # Return the output object
    return output
