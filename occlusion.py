# setup the environment
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy
from scipy import misc
from scipy import ndimage
from PIL import Image
import cPickle as pickle
from scipy.stats.stats import pearsonr
import os
import sys

if len(sys.argv) > 1:
    fname = sys.argv[1]
else:
    print "Require a file to store the output data"
    exit(0)
    
# set the defaults for display
plt.rcParams['figure.figsize'] = (10,10)   # image size
plt.rcParams['image.interpolation'] = 'nearest' # show square pixels, don't interpolate
plt.rcParams['image.cmap'] = 'gray' # use grayscale output rather than a colour heat map (which can be misleading)


# Load Caffe
import caffe

# Load the net

# get the path to caffe root
caffe_root = '../'

# define the model structure
model_def = caffe_root + 'scene/placesCNN_upgraded/places205CNN_deploy_upgraded.prototxt'

# get the trained weights
model_weights = caffe_root + 'scene/placesCNN_upgraded/places205CNN_iter_300000_upgraded.caffemodel'

# build the net, use the test mode (e.g., no dropout)
net = caffe.Net(model_def, model_weights, caffe.TEST)

# Input preprocessing
# load the mean places image (generated using the convert_protomean.py script)
mi = np.load(caffe_root + 'scene/placesCNN_upgraded/places205CNN_mean.npy')
print "Mean Image Shape: ", mi.shape
mi = mi.mean(1).mean(1)  #avg over pixels to get the mean (BGR) pixel values
print "Mean-subtracted values: ", zip('BGR', mi)

# create a transformer called data for the input 
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
# move image channels to outermost dimension
transformer.set_transpose('data', (2,0,1)) 
# In each channel, subtract the dataset value
transformer.set_mean('data', mi)
# rescale from [0,1] to [0,255]
transformer.set_raw_scale('data', 255)
# swap the channels from RGB to BGR
transformer.set_channel_swap('data', (2,1,0))


## GPU classification
caffe.set_device(0)  # pick the first GPU in case of multiple gpus
caffe.set_mode_gpu()
GPUstart_time = time.clock()


# set the size of the input if different from the default
net.blobs['data'].reshape(1,  # batch size
                           3,  # 3-channel BGR images
                           227, 227)  # image size is 227x227
                           
# Load the labels for the places database
labels_file = caffe_root + 'scene/placesCNN_upgraded/categoryIndex_places205.csv'
labels = np.loadtxt(labels_file, str, delimiter='\t') 



def occlusion_heatmap(net, x, target, square_length=7):
    """An occlusion test that checks an image for its critical parts.
    In this function, a square part of the image is occluded (i.e. set
    to 0) and then the net is tested for its propensity to predict the
    correct label. One should expect that this propensity shrinks of
    critical parts of the image are occluded. If not, this indicates
    overfitting.
    Depending on the depth of the net and the size of the image, this
    function may take awhile to finish, since one prediction for each
    pixel of the image is made.
    Currently, all color channels are occluded at the same time. Also,
    this does not really work if images are randomly distorted by the
    batch iterator.
    See paper: Zeiler, Fergus 2013
    Parameters
    ----------
    net : NeuralNet instance
      The neural net to test.
    x : np.array
      The input data, should be of shape (K, H, W) where K is the no. of channels. 
      H is hegith and W is the width of the image (i.e., number of pixels).
      This will be the transformed_image in caffe.
    target : int
      The true value of the image (label_index). If the net makes several
      predictions, say 205 classes, this indicates which one to look
      at.
    square_length : int (default=7)
      The length of the side of the square that occludes the image.
      Must be an odd number.
    Results
    -------
    heat_array : np.array (with same size as image)
      An 2D np.array that at each point (i, j) contains the predicted
      probability of the correct class if the image is occluded by a
      square with center (i, j).
    """
    
    if x.ndim != 3:
        raise ValueError("This function requires the input data to be of "
                         "shape (K, H, W), instead got {}".format(x.shape))
                         
    # odd length squares needed since we need a center pixel
    if square_length % 2 == 0:
        raise ValueError("Square length has to be an odd number, instead "
                         "got {}.".format(square_length))
                         
    
    c, h, w = x.shape       # get the dimensions of the image             
    heat_array = np.zeros((w, h))
    img = x.copy()   # copy the image
    pad = square_length // 2 
    occluded_images = np.zeros((227, c, w, h), dtype=img.dtype)

    # generate occluded images
    for i in range(w):
        # batch h occluded images for speeding up predictions
        for j in range(h):
            # default constant value is 0, so padding with zeros
            x_pad = np.pad(img, ((0, 0), (pad, pad), (pad, pad)), 'constant')
            # set the values in the square to zero
            x_pad[:, i:i + square_length-1, j:j + square_length-1] = 0
   
            # x_pad is the new image that needs to be passed through the net         
            # but our input image size is 227*227 so extract the occluded image
            # of the size of original image. Occlusion region is centered at i,j
            occluded_images[j] = x_pad[:, pad:pad+h, pad:pad+w]
            
     

        # set the size of the input if different from the default
        net.blobs['data'].reshape(114,  # batch size
                                    3,  # 3-channel BGR images
                                227, 227)  # image size is 227x227

        # Classify the image
        # copy the image data to the memory allocated for the conv net
        net.blobs['data'].data[...] = occluded_images[0:114,:,:,:]         
            
        # perform classification
        output = net.forward()
        output_prob = output['prob']
        for j in range(114):    
            corr_prob = output_prob[j].item(target)   # probability of true value
            heat_array[i,j] = corr_prob


        # set the size of the input if different from the default
        net.blobs['data'].reshape(113,  # batch size
                                    3,  # 3-channel BGR images
                                227, 227)  # image size is 227x227

        net.blobs['data'].data[...] = occluded_images[114:227,:,:,:]       
        # perform classification
        output = net.forward()
        output_prob = output['prob']
        for j in range(113):    
            corr_prob = output_prob[j].item(target)   # probability of true value
            heat_array[i,j+114] = corr_prob

            
    return heat_array


## Load the image and generate a heat map
## load the image
#db_name = caffe_root + 'scene/placesCNN_upgraded/testSet_resize/'
db_name = caffe_root + 'examples/test/'
label_index = 24

i=0
for filename in os.listdir(db_name):
    filename = 'bedroom.jpg'
    image = caffe.io.load_image(db_name + filename)
  
    transformed_image = transformer.preprocess('data', image)
    #plt.figure()
    #plt.imshow(image)
    #plt.show()
  
    # create and plot a heatmap
    heat_array = occlusion_heatmap(net, transformed_image, label_index, square_length=3)
    
    # store the data in a pickle file for easy plot reconstruction
    # this helps to view the plot without having to run the script again
    data = {
            'heat_array': heat_array,
            'transformed_image': transformed_image,
            }
    
    # fname comes from arguments
    pickle.dump(data, open(fname, 'wb'))
    print ("pickle complete")
    print (fname)
        
    if i==0:
        break
      
execution_time = time.clock() - GPUstart_time
print "Execution time in GPU mode: ", execution_time, " seconds or ", execution_time/60.0 , " minutes"
