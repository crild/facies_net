# -------- training data creator ---------

# import prerequisite modules
import numpy as np

# Function that creates .pts files for training
def data_creator(corner_coords,sample_rates,save_name):
    # corner_coordinates: cordinates for the top left and bottom right points
    # sample_rate: inline, xline, time sample rates

    # Define individual step lengths
    inline_step = sample_rates[0]
    xline_step = sample_rates[1]
    time_step = sample_rates[2]

    # Define total section counter:
    sec_len = 0

    # Make sure the corner coordinates are within pixels
    corner_coords[:,:,0] = np.floor(corner_coords[:,:,0]/inline_step)*inline_step
    corner_coords[:,:,1] = np.floor(corner_coords[:,:,1]/xline_step)*xline_step
    corner_coords[:,:,2] = np.floor(corner_coords[:,:,2]/time_step)*time_step

    arr = np.empty((0,3))

    for corner_coord in corner_coords:
        # Define some limiting parameters
        inline_length = int((corner_coord[1,0]-corner_coord[0,0])//inline_step)+1
        xline_length = int((corner_coord[1,1]-corner_coord[0,1])//xline_step)+1
        time_length = int((corner_coord[1,2]-corner_coord[0,2])//time_step)+1


        inl_range = np.arange(corner_coord[0,0],corner_coord[1,0]+inline_step,inline_step)
        xl_range = np.arange(corner_coord[0,1],corner_coord[1,1]+xline_step,xline_step)
        t_range = np.arange(corner_coord[0,2],corner_coord[1,2]+time_step,time_step)

        out_array = np.empty((inline_length*xline_length*time_length,3))

        for inl in inl_range:
            i = int((inl - corner_coord[0,0])//inline_step)
            for xl in xl_range:
                j = int((xl - corner_coord[0,1])//xline_step)
                for tim in t_range:
                    k = int((tim - corner_coord[0,2])//time_step)
                    out_array[i*xline_length*time_length+j*time_length+k] = [inl,xl,tim]

        arr = np.append(arr,out_array,axis = 0)

    # Save csv file
    np.savetxt(fname = save_name+'.pts',X = arr,fmt = '%i')


# Define the function parameters
top_left1 = [6960,40858,1912]
bot_right1 = [8014,40858,3886]

top_left2 = [7080,40858,566]
bot_right2 = [7604,40858,1098]

samp_rt = np.array([1,1,4])

name = 'Other_ilxl'

# Set the parameters correctly and run the function
c_cord = np.array([[top_left1,bot_right1],
                  [top_left2,bot_right2]])


data_creator(c_cord,samp_rt,name)
