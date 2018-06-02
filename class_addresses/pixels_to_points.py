# -------- pixels to points ---------

# import prerequisite modules
import numpy as np


# Function that converts UTM coordinates to il,xl
def ilxl_to_utm(reference_pts):
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
def pixels_to_points(filename,sample_rates,ref_points,save_name):
    # corner_coordinates: cordinates for the top left and bottom right points
    # sample_rate: inline, xline, time sample rates

    # Define individual step lengths
    inline_step = sample_rates[0]
    xline_step = sample_rates[1]
    time_step = sample_rates[2]

    # Get the UTM to il,xl factors
    (a_q,b_q) = ilxl_to_utm(ref_points)

    a = np.loadtxt(filename, skiprows=0, usecols = range(3), dtype = np.float32)


    X = a[:,0]*a_q[0]+a[:,1]*a_q[1]+a_q[2];
    Y = a[:,0]*b_q[0]+a[:,1]*b_q[1]+b_q[2];
    time = a[:,2]

    # Append the examples to the right list
    arr = np.array([X,Y,time])

    # Save csv file
    np.savetxt(fname = save_name+'.pts',X = arr.T,fmt = '%f')

# Define the function parameters
samp_rt_Hoop = np.array([1,1,4])


# Give input coordinates first, then output coordinates
# Hoop il,xl, X,Y
r1_Hoop=[6650,38871,460998.02,8170404.19];
r2_Hoop=[9755,38871,494923.54,8254372.33];
r3_Hoop=[6650,46116,514976.54,8148595.28];

# Hoop X,Y, il,xl
r1_Hoop_2=[460998.02,8170404.19,6650,38871];
r2_Hoop_2=[494923.54,8254372.33,9755,38871];
r3_Hoop_2=[514976.54,8148595.28,6650,46116];

# F3 il,xl, X,Y
r1_F3=[100,300,605835.50,6073556.50];
r2_F3=[750,300,629576.52,6074220.01];
r3_F3=[100,1250,605381.54,6089799.49];

# F3 X,Y, il,xl
r1_F3_2=[605835.50,6073556.50,100,300];
r2_F3_2=[629576.52,6074220.01,750,300];
r3_F3_2=[605381.54,6089799.49,100,1250];

ref_list = np.array([r1_Hoop_2,r2_Hoop_2,r3_Hoop_2])



filename = 'test1_ilxl.pts'
name = 'test1_utm'

pixels_to_points(filename,samp_rt_Hoop,ref_list,name)
