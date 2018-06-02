# -------- points to pixels ---------

# import prerequisite modules
import numpy as np

# Function that converts UTM coordinates to il,xl
def utm_to_ilxl(reference_pts):
    # reference_pts: array of point(at least 3) with both UTM and il,xl corner_coordinates

    if reference_pts.shape[0] > 3:
        ind1 = np.argmin(reference_pts[:,0])
        ind2 = np.argmax(reference_pts[:,0])
        ind3 = np.argmin(reference_pts[:,1])
        mm = np.array([reference_pts[ind1,:],reference_pts[ind2,:],reference_pts[ind3,:]])
    else:
        mm = reference_pts

    M = np.array([[mm[0,0],mm[0,1],1],[mm[1,0],mm[1,1],1],[mm[2,0],mm[2,1],1]])

    if np.linalg.det(M) == 0:
        print('Chosen points give no unique solution')
    else:
        a = np.linalg.solve(M,np.array([mm[0,2],mm[1,2],mm[2,2]]))
        b = np.linalg.solve(M,np.array([mm[0,3],mm[1,3],mm[2,3]]))

    return a, b

# Function that creates .pts files for training data from UTM data
def points_to_pixels(filename,sample_rates,ref_points,save_name):
    # corner_coordinates: cordinates for the top left and bottom right points
    # sample_rate: inline, xline, time sample rates

    # Define individual step lengths
    inline_step = sample_rates[0]
    xline_step = sample_rates[1]
    time_step = sample_rates[2]

    # Get the UTM to il,xl factors
    (a_q,b_q) = utm_to_ilxl(ref_points)

    a = np.loadtxt(filename, skiprows=0, usecols = range(3), dtype = np.int32)


    il = a[:,0]*a_q[0]+a[:,1]*a_q[1]+a_q[2];
    xl = a[:,0]*b_q[0]+a[:,1]*b_q[1]+b_q[2];

    # round off to closest pixel
    il = np.floor(il/inline_step)*inline_step
    xl = np.floor(xl/xline_step)*xline_step
    time = np.floor(a[:,2]/time_step)*time_step


    # Append the examples to the right list
    arr = np.array([il,xl,time])

    # Save csv file
    np.savetxt(fname = save_name+'.pts',X = arr.T,fmt = '%i')


# Define the function parameters
samp_rt_Hoop = np.array([1,1,4])

# UTM_il,UTM_xl, il, xl
r1_Hoop=[460998.02,8170404.19,6650,38871];
r2_Hoop=[494923.54,8254372.33,9755,38871];
r3_Hoop=[514976.54,8148595.28,6650,46116];

ref_pts_Hoop = np.array([r1,r2,r3])


filename = 'Snadd.pts'
name = 'Snadd_ilxl'

points_to_pixels(filename,samp_rt_Hoop,ref_pts_Hoop,name)
