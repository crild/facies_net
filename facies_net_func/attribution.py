################################################################
# Implemented by Naozumi Hiranuma (hiranumn@uw.edu)            #
#                                                              #
# Keras-compatible implmentation of Integrated Gradients       #
# proposed in "Axiomatic attribution for deep neuron networks" #
# (https://arxiv.org/abs/1703.01365).                          #
#                                                              #
# Keywords: Shapley values, interpretable machine learning     #
################################################################

from __future__ import division, print_function

import sys
import keras.backend as K
import numpy as np

from scipy.misc import imsave
from time import sleep

from keras.models import Model, Sequential

'''
Integrated gradients approximates Shapley values by integrating partial
gradients with respect to input features from reference input to the
actual input. The following class implements the paper "Axiomatic attribution
for deep neuron networks".
'''
class integrated_gradients:
    # model: Keras model that you wish to explain.
    # outchannels: In case the model are multi tasking, you can specify which output you want explain .
    def __init__(self, model, outchannels=[], verbose=1):

        #get backend info (either tensorflow or theano)
        self.backend = K.backend()

        #load model supports keras.Model and keras.Sequential
        if isinstance(model, Sequential):
            self.model = model.model
        elif isinstance(model, Model):
            self.model = model
        else:
            print("Invalid input model")
            return -1

        #load input tensors
        self.input_tensors = []
        for i in self.model.inputs:
            self.input_tensors.append(i)
        # The learning phase flag is a bool tensor (0 = test, 1 = train)
        # to be passed as input to any Keras function that uses
        # a different behavior at train time and test time.
        self.input_tensors.append(K.learning_phase())

        #If outputchanels are specified, use it.
        #Otherwise evalueate all outputs.
        self.outchannels = outchannels
        if len(self.outchannels) == 0:
            if verbose: print("Evaluated output channel (0-based index): All")
            if K.backend() == "tensorflow":
                self.outchannels = range(self.model.output_shape[1])
            elif K.backend() == "theano":
                self.outchannels = range(self.model.output._keras_shape[1])
        else:
            if verbose:
                print("Evaluated output channels (0-based index):")
                print(','.join([str(i) for i in self.outchannels]))

        #Build gradient functions for desired output channels.
        self.get_gradients = {}
        if verbose: print("Building gradient functions")

        # Evaluate over all requested channels.
        for c in self.outchannels:
            # Get tensor that calculates gradient
            gradients = self.model.optimizer.get_gradients(self.model.output[:, c], self.model.input)

            # Build computational graph that computes the tensors given inputs
            self.get_gradients[c] = K.function(inputs=self.input_tensors, outputs=gradients)

            # This takes a lot of time for a big model with many tasks.
            # So lets print the progress.
            if verbose:
                sys.stdout.write('\r')
                sys.stdout.write("Progress: "+str(int((c+1)*1.0/len(self.outchannels)*1000)*1.0/10)+"%")
                sys.stdout.flush()
        # Done
        if verbose: print("\nDone.")


    '''
    Input: sample to explain, channel to explain
    Optional inputs:
        - reference: reference values (defaulted to 0s).
        - steps: # steps from reference values to the actual sample (defualted to 50).
    Output: list of numpy arrays to integrated over.
    '''
    def explain(self, sample, outc=0, reference=False, num_steps=50, verbose=0):

        # Each element for each input stream.
        samples = []
        numsteps = []
        step_sizes = []

        # If multiple inputs are present, feed them as list of np arrays.
        if isinstance(sample, list):
            #If reference is present, reference and sample size need to be equal.
            if reference != False:
                assert len(sample) == len(reference)
            for i in range(len(sample)):
                if reference == False:
                    _output = integrated_gradients.linearly_interpolate(sample[i], False, num_steps)
                else:
                    _output = integrated_gradients.linearly_interpolate(sample[i], reference[i], num_steps)
                samples.append(_output[0])
                numsteps.append(_output[1])
                step_sizes.append(_output[2])

        # Or you can feed just a single numpy arrray.
        elif isinstance(sample, np.ndarray):
            _output = integrated_gradients.linearly_interpolate(sample, reference, num_steps)
            samples.append(_output[0])
            numsteps.append(_output[1])
            step_sizes.append(_output[2])

        # Desired channel must be in the list of outputchannels
        assert outc in self.outchannels
        if verbose: print("Explaining the "+str(self.outchannels[outc])+"th output.")

        # For tensorflow backend
        _input = []
        for s in samples:
            _input.append(s)
        _input.append(0)

        gradients = self.get_gradients[outc](_input)

        explanation = []
        for i in range(len(gradients)):
            _temp = np.sum(gradients[i], axis=0)
            explanation.append(np.multiply(_temp, step_sizes[i]))

        # Format the return values according to the input sample.
        if isinstance(sample, list):
            return explanation
        elif isinstance(sample, np.ndarray):
            return explanation[0]
        return -1


    '''
    Input: numpy array of a sample
    Optional inputs:
        - reference: reference values (defaulted to 0s).
        - steps: # steps from reference values to the actual sample.
    Output: list of numpy arrays to integrate over.
    '''
    @staticmethod
    def linearly_interpolate(sample, reference=False, num_steps=50):
        # Use default reference values if reference is not specified
        if reference is False: reference = np.zeros(sample.shape);

        # Reference and sample shape needs to match exactly
        assert sample.shape == reference.shape

        # Calcuated stepwise difference from reference to the actual sample.
        ret = np.zeros(tuple([num_steps] +[i for i in sample.shape]))
        for s in range(num_steps):
            ret[s] = reference+(sample-reference)*(s*1.0/num_steps)

        return ret, num_steps, (sample-reference)*(1.0/num_steps)


# Format processing of image
def form_pros(im,formatting = 'normalize',clim = np.array([0, 255])):
    # im: image to process,
    # formatting: convention of formatting to use

    # Process the images
    if formatting is None:
        im2 = im

    elif formatting == 'normalize':
        # normalize tensor: center on 0., ensure std is 0.1
        mn = im.mean()
        stad = im.std()
        im1 = ((im-mn)/(stad+1e-10))*0.1
        fac = clim[1]-clim[0]

        # cast to rgb value range
        im2 = np.clip(np.clip(im1+0.5, 0, 1)*fac+clim[0],clim[0],clim[1]).astype('uint8')

    elif formatting == 'RGBcast':
        # cast to [0, 255]
        maxima = np.amax(im)
        minima = np.amin(im)
        interv = maxima - minima
        im1 = im - minima
        im2 = (im1/interv)*(clim[1]-clim[0])+clim[0]

    else:
        print('Illegal formatting string!')

    return im2


# Function to lay one image over the other to show attribution better
def overlay(or_im, overlay_im, mode = 'red'):
    # or_im: Orginal image to use as background
    # overlay_im: image to use as the overlay
    # mode: what type of overlay to use (red-scale,opacity, etc.)

    # Get the image parts of the arrays
    or_cube = or_im[0,:,:,:,0]
    ol_cube = overlay_im[0,:,:,:,0]

    # reformat image to optimize for scipy's expectations
    or_cube = form_pros(or_cube)

    # Check if the user is using multi-channel analysis
    if mode == 'RB':
        ol_cube_red = np.copy(ol_cube)
        ol_cube_blue = np.copy(ol_cube)

        cut_off = np.std(ol_cube)
        ol_cube_red[ol_cube <= cut_off] = 0
        ol_cube_blue[ol_cube >= -cut_off] = 0

        ol_cube_blue = np.absolute(ol_cube_blue)
        ol_cube_red_adj = form_pros(ol_cube_red)
        ol_cube_blue_adj = form_pros(ol_cube_blue)

        ol_re = np.copy(or_cube)
        ol_gr = np.copy(or_cube)
        ol_bl = np.copy(or_cube)

        ol_re[ol_cube_red>0] = ol_cube_red_adj[ol_cube_red>0]
        ol_bl[ol_cube_red>0] = 0

        ol_re[ol_cube_blue>0] = 0
        #ol_gr[ol_cube_blue>0] = 0 #[0.5*ol_gr[ol_cube_blue>0]].astype('uint8')
        ol_bl[ol_cube_blue>0] = ol_cube_blue_adj[ol_cube_blue>0]

        Re = ol_re
        Gr = ol_gr
        Bl = ol_bl

        output_im = np.stack([Re,Gr,Bl], axis=-1)

    elif mode == 'opacity':
        ol_cube = form_pros(ol_cube)
        output_im = np.stack([or_cube,or_cube,or_cube,ol_cube], axis=-1)

    elif mode == 'red':
        ol_cube = form_pros(ol_cube)
        output_im = np.stack([ol_cube,or_cube,or_cube], axis=-1)

    elif mode == 'green':
        ol_cube = form_pros(ol_cube)
        output_im = np.stack([or_cube,ol_cube,or_cube], axis=-1)

    elif mode == 'blue':
        ol_cube = form_pros(ol_cube)
        output_im = np.stack([or_cube,or_cube,ol_cube], axis=-1)

    else:
        print('Uknown mode! Printing original image.')
        output_im = np.stack([or_cube,or_cube,or_cube], axis=-1)

    return output_im





def save_overlay(int_grad,classes,inp_im,name=None,steps = 100, mosaic = 'cols'):
    # or_im: Orginal image to use as background
    # overlay_im: image to use as the overlay
    # mode: what type of overlay to use (red-scale,opacity, etc.)
    # name: filename of the saved image

    # Make the empty stiched image
    if mosaic == 'cols':
        # Define some initial parameters
        margin = 5
        width = classes * 61 + (classes - 1) * margin
        height = (2*3) * 61 + ((2*3) - 1) * margin

    elif mosaic == 'rows':
        # Define some initial parameters
        margin = 5
        width = (2*3) * 61 + ((2*3) - 1) * margin
        height = (classes) * 61 + ((classes) - 1) * margin

    elif mosaic == 'rows2':
        # Define some initial parameters
        margin = 5
        width = 3 * 61 + (3 - 1) * margin
        height = (2*classes) * 61 + ((2*classes) - 1) * margin

    else:
        print('Unknown mosaic input.')
        return


    # Put it together to make the image
    stitched_im = np.zeros((height,width,3))

    for i in range(classes):
        # Make the explanation image
        explanation = np.expand_dims(int_grad.explain(inp_im[0],outc=i,num_steps=steps,verbose=1),axis=0)

        output_im = form_pros(explanation,formatting='normalize')

        # Iterate through the directions of the cube
        for k in range(3):
            # slice the 3D input into 2.5D (one slice from each plane)
            if k == 0:
                im = output_im[0,30,:,:,:]
            elif k == 1:
                im = output_im[0,:,30,:,:]
            elif k == 2:
                im = output_im[0,:,:,30,:]
            else:
                print('Undefined scenario!')

            # Make stratigraphic upwards direction up in the image
            im = np.transpose(im,(1,0,2))

            if mosaic == 'cols':
                # Add it to the stitched image
                stitched_im[(61 + margin) * k:(61 + margin) * k + 61,(61 + margin) * i: (61 + margin) * i + 61,:] = im
            elif mosaic == 'rows':
                # Add it to the stitched image
                stitched_im[(61 + margin) * i: (61 + margin) * (i) + 61,(61 + margin) * k:(61 + margin) * k + 61,:] = im
            elif mosaic == 'rows2':
                # Add it to the stitched image
                stitched_im[(61 + margin) * (2*i): (61 + margin) * (2*i) + 61,(61 + margin) * k:(61 + margin) * k + 61,:] = im



        # Overlay the explanation over the initial image
        output_im = overlay(inp_im,explanation,mode = 'RB')

        # Iterate through the directions of the cube
        for k in range(3):
            # slice the 3D input into 2.5D (one slice from each plane)
            if k == 0:
                im = output_im[30,:,:,:]
            elif k == 1:
                im = output_im[:,30,:,:]
            elif k == 2:
                im = output_im[:,:,30,:]
            else:
                print('Undefined scenario!')

            # Make stratigraphic upwards direction up in the image
            im = np.transpose(im,(1,0,2))

            if mosaic == 'cols':
                # Add it to the stitched image
                stitched_im[(61 + margin) * (k + 3):(61 + margin) * (k + 3) + 61,(61 + margin) * i: (61 + margin) * i + 61,:] = im
            elif mosaic == 'rows':
                # Add it to the stitched image
                stitched_im[(61 + margin) * (i): (61 + margin) * (i) + 61,(61 + margin) * (k + 3):(61 + margin) * (k + 3) + 61,:] = im
            elif mosaic == 'rows2':
                # Add it to the stitched image
                stitched_im[(61 + margin) * (2*i + 1): (61 + margin) * (2*i + 1) + 61,(61 + margin) * k:(61 + margin) * k + 61,:] = im


    # save the result to disk
    if name is None:
        imsave('Overlay_im.png', stitched_im)
        print('file name is: ','Overlay_im.png')
    else:
        name += '.png'
        imsave(name, stitched_im)
        print('file name is: ',name)

    return stitched_im
