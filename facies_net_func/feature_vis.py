import keras
import numpy as np
import time

from keras import backend as K
from scipy.misc import imsave
from scipy.ndimage import filters


# util function to convert a tensor into a valid image
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def normalize(x):
    # utility function to normalize a tensor by a given norm (standard is L2 norm)
    return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())


def smoothing(im, mode = None):
    # utility function to smooth an image
    if mode is None:
        return im
    elif mode == 'L2':
        # L2 norm
        return im / (np.sqrt(np.mean(np.square(im))) + K.epsilon())
    elif mode == 'GaussianBlur':
        # Gaussian Blurring with width of 3
        return filters.gaussian_filter(im,1/8)
    elif mode == 'Decay':
        # Decay regularization
        decay = 0.98
        return decay * im
    elif mode == 'Clip_weak':
        # Clip weak pixel regularization
        percentile = 1
        threshold = np.percentile(np.abs(im),percentile)
        im[np.where(np.abs(im) < threshold)] = 0
        return im
    else:
        # print error message
        print('Unknown smoothing parameter. No smoothing implemented.')
        return im


def save_image(kept_filters, keras_model, name = None):
    # kept_filters: list of inputs that give the highest response from each node
    # keras_model: trained model that we are visualizing

    # dimensions of the generated pictures for each filter.
    input_shape = keras_model.input_shape
    img_width = input_shape[1]
    img_height = input_shape[2]
    img_depth = input_shape[3]

    # the filters that have the highest loss are assumed to be better-looking.
    # we will only keep the top 24 filters if we have more than 24.
    kept_filters.sort(key=lambda x: x[1], reverse=True)
    kept_filters = kept_filters[:24]

    if (len(kept_filters) <= 4):
        # we will stich the best 64 filters on a 1 x n grid.
        nj = 1 # height
        ni = len(kept_filters)//nj # width
        nk = 3 # depth

    elif (len(kept_filters) > 4) and (len(kept_filters) < 24):
        # we will stich the best filters on a 4 x n grid.
        nj = 4 # height
        ni = len(kept_filters)//nj # width
        nk = 3 # depth

    else:
        # we will stich the best 24 filters on a 8 x 9 grid.
        nj = 8 # height
        ni = len(kept_filters)//nj # width
        nk = 3 # depth


    # build a black picture with enough space for
    # our 8 x 9 filters of size 61 x 61, with a 5px margin in between
    margin = 5
    width = (ni*nk) * img_width + (ni*nk - 1) * margin
    height = nj * img_height + (nj - 1) * margin
    stitched_filters = np.zeros((height,width, 3))

    # fill the picture with our saved filters
    for i in range(ni):
        for j in range(nj):
            img, loss = kept_filters[i * nj + j]
            for k in range(nk):
                # slice the 3D input into 2.5D (one slice from each plane)
                if k == 0:
                    im = np.transpose(img[30,:,:,:],(1,0,2))
                elif k == 1:
                    im = np.transpose(img[:,30,:,:],(1,0,2))
                elif k == 2:
                    im = np.transpose(img[:,:,30,:],(1,0,2))
                else:
                    print('Undefined scenario!')

                stitched_filters[(img_height + margin) * j: (img_height + margin) * j + img_height,
                                 (img_width + margin) * (i*nk + k): (img_width + margin) * (i*nk + k) + img_width,:] = im

    # save the result to disk
    if name is None:
        imsave('stitched_filters_%dx%d.png' % (nj,ni*nk), stitched_filters)
        print('file name is: ','stitched_filters_%dx%d.png' % (nj,ni*nk))
    else:
        name += '.png'
        imsave(name, stitched_filters)
        print('file name is: ',name)



# save the original image to disk
def save_or(img,name = None,formatting = 'normalize'):
    # img: image to be saved to disk

    # Define some initial parameters
    input_channels = 3
    margin = 5
    width = 3 * 61 + (3 - 1) * margin
    height = 61 + (1 - 1) * margin

    stitched_im = np.zeros((height,width,input_channels))

    # Iterate through the directions of the cube
    for k in range(3):
        # slice the 3D input into 2.5D (one slice from each plane)
        if k == 0:
            im = img[0,30,:,:,:]
        elif k == 1:
            im = img[0,:,30,:,:]
        elif k == 2:
            im = img[0,:,:,30,:]
        else:
            print('Undefined scenario!')

        # Make stratigraphic upwards direction up in the image
        im = np.transpose(im,(1,0,2))

        # Process the images
        if formatting is None:
            im2 = im

        elif formatting == 'normalize':
            # normalize tensor: center on 0., ensure std is 0.1
            mn = im.mean()
            stad = im.std()
            im1 = ((im-mn)/(stad+1e-10))*0.1

            # cast to rgb value range
            im2 = np.clip(np.clip(im1+0.5, 0, 1)*255,0,255).astype('uint8')

        elif formatting == 'RGBcast':
            # cast to [0, 255]
            maxima = np.amax(im)
            minima = np.amin(im)
            interv = maxima - minima
            im1 = im - minima
            im2 = (im1/interv)*255

        else:
            print('Illegal formatting string!')

        stitched_im[0:61,(61 + margin) * k: (61 + margin) * k + 61,:] = im2

    # save the result to disk
    if name is None:
        imsave('Original_im.png', stitched_im)
        print('file name is: ','Original_im.png')
    else:
        name += '.png'
        imsave(name, stitched_im)
        print('file name is: ',name)



def features(model,layer_name,iterations = 50,smoothing_par = 'GaussianBlur',
             inp_im = None, name = None):
    # model: model to visualize
    # layer_name: name of the layer we want to visualize features of
    # smoothing: smoothing function

    # Give a model summary to ensure it is correctly uploaded
    model.summary()

    # dimensions of the generated pictures for each filter.
    input_shape = model.input_shape
    img_width = input_shape[1]
    img_height = input_shape[2]
    img_depth = input_shape[3]

    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])

    # this is the placeholder for the input images
    input_img = model.input

    # find out how many filters we need to calculate for
    num_filts = layer_dict[layer_name].output_shape[-1]

    # Iterate through the filters to find the optimal inputs
    kept_filters = []
    loss_list = np.empty((0,3), dtype = np.float32)
    for filter_index in range(num_filts):
        # Print a message to the user and start a timer
        print('Processing filter %d' % filter_index)
        start_time = time.time()

        # we build a loss function that maximizes the activation
        # of the nth filter of the layer considered
        layer_output = layer_dict[layer_name].output
        loss = K.mean(layer_output[:, :, :, :, filter_index])

        # we compute the gradient of the input picture wrt this loss
        grads = K.gradients(loss, input_img)[0]

        # normalization trick: we normalize the gradient
        grads = normalize(grads)

        # this function returns the loss and grads given the input picture
        iterate = K.function([input_img], [loss, grads])

        # step size for gradient ascent
        step = 0.9

        # Make a random start-point for the optimization if the user hasn`t define one
        if inp_im is None:
            # we start from a gray image with some random noise
            input_img_data = np.random.random((1, img_width, img_height, img_depth, 1))
            input_img_data = (input_img_data - 0.5) * 25    # clim is [-127 127]
        else:
            input_img_data = inp_im

        # we run gradient ascent for m steps
        for i in range(iterations):
            loss_value, grads_value = iterate([input_img_data])
            input_img_data += grads_value * step

            # For every improvement except the last we perform smoothing
            if i < iterations-1:
                input_img_data = smoothing(input_img_data,mode = smoothing_par)

            # Give an update of the loss to the user 5 times over the span of optimization
            if i % (iterations//5) == 0:
                print('Loss value at iteration %d:' %i, loss_value)

                if i == 0:
                    init_loss = loss_value

        # decode the resulting input image and append it to the lists
        loss_list = np.append(loss_list,[[loss_value, filter_index, init_loss]],axis = 0)
        if loss_value > 0:
            img = deprocess_image(input_img_data[0])
            kept_filters.append((img, loss_value))
        end_time = time.time()
        print('Filter %d processed in %ds' % (filter_index, end_time - start_time))

    # Make the mosaic and save it as an image
    save_image(kept_filters, model, name)

    # Sort the list of filter losses so the user knows which filters were best
    sort_idx = loss_list.argsort(axis = 0)
    sort_idx = sort_idx[:,0]


    return kept_filters, loss_list[sort_idx[::-1]]
