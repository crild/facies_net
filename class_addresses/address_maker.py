# -------- training data creator ---------

# import prerequisite modules
import numpy as np

# Function that creates .pts files for training
def data_creator(corner_coord,sample_rates,save_name):
    # corner_coordinates: cordinates for the top left and bottom right points
    # sample_rate: inline, xline, time sample rates

    # Define individual step lengths
    inline_step = sample_rates[0]
    xline_step = sample_rates[1]
    time_step = sample_rates[2]

    inline_length = int((corner_coord[1,0]-corner_coord[0,0])//inline_step)
    xline_length = int((corner_coord[1,1]-corner_coord[0,1])//xline_step)
    time_length = int((corner_coord[1,2]-corner_coord[0,2])//time_step)


    inl_range = np.arange(corner_coord[0,0],corner_coord[1,0]+inline_step,inline_step)
    xl_range = np.arange(corner_coord[0,1],corner_coord[1,1]+xline_step,xline_step)
    t_range = np.arange(corner_coord[0,2],corner_coord[1,2]+time_step,time_step)

    out_array = np.empty(((inline_length+1)*(xline_length+1)*(time_length+1),3))

    for inl in inl_range:
        i = int((inl - corner_coord[0,0])//inline_step)
        for xl in xl_range:
            j = int((xl - corner_coord[0,1])//xline_step)
            for tim in t_range:
                k = int((tim - corner_coord[0,2])//time_step)
                out_array[i*inline_length+j*xline_length+k] = [inl,xl,tim]

    # Save csv file
    np.savetxt(fname = save_name+'.pts',X = out_array,fmt = '%f')


# Define the function parameters
top_left = [100,200,44.4]
bot_right = [106,200,52.4]

samp_rt = np.array([2,2,4])

name = 'test'

# Set the parameters correctly and run the function
c_cord = np.array([top_left,bot_right])
data_creator(c_cord,samp_rt,name)
