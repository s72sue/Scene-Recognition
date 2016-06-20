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
np.set_printoptions(threshold='nan')

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
#caffe.set_mode_gpu()
GPUstart_time = time.clock()


# set the size of the input if different from the default
net.blobs['data'].reshape(1,  # batch size
                           3,  # 3-channel BGR images
                           227, 227)  # image size is 227x227

num_images = 100
num_layers = 8

lconv1_activations = np.zeros((96*55*55, num_images))
lconv2_activations = np.zeros((256*27*27, num_images))
lconv3_activations = np.zeros((384*13*13, num_images))
lconv4_activations = np.zeros((384*13*13, num_images))
lconv5_activations = np.zeros((256*13*13, num_images))
lfc6_activations = np.zeros((4096, num_images))
lfc7_activations = np.zeros((4096, num_images))
lfc8_activations = np.zeros((205, num_images))

rconv1_activations = np.zeros((96*55*55, num_images))
rconv2_activations = np.zeros((256*27*27, num_images))
rconv3_activations = np.zeros((384*13*13, num_images))
rconv4_activations = np.zeros((384*13*13, num_images))
rconv5_activations = np.zeros((256*13*13, num_images))
rfc6_activations = np.zeros((4096, num_images))
rfc7_activations = np.zeros((4096, num_images))
rfc8_activations = np.zeros((205, num_images))



figure_seq = []
def get_activations(dir_name, label):
    i = 0
    flag = 0
    # go over the image segments in the given directory
    # this might be right, left segments or the full image
    if label == "full":
        image_list = os.listdir(dir_name)
        flag = 0
    else:
        image_list = figure_seq  
        flag = 1  # to distinguish between left/right vs directory names

    for filename in image_list:
        name = filename.split(".")
        if len(name) < 2 and flag == 0: #or name[0][0:4] != 'gard':
            continue
        elif label == "full":
            figure_seq.append(name[0])
            image_file = filename
        elif label == "right":
            image_file = filename + '_right.jpg'
        else:
            image_file = filename + '_left.jpg'

        # figure out the category of the image
        #name = filename.split(".")
        #name = name[0][0:4]  # get the first four letters of the filename
        #category = category_seq.index(name) 
        
        # load an image and perform the preprocessing
        image = caffe.io.load_image(dir_name + image_file)
        #print dir_name + image_file
    
        transformed_image = transformer.preprocess('data', image)
        #print "TEST22"
        #plt.figure()
        #plt.imshow(image)
        #plt.show()
    
        # Classify the image
        # copy the image data to the memory allocated for the conv net
        net.blobs['data'].data[...] = transformed_image
        #print "test 44"
    
        # perform classification
        output = net.forward()
        #print "Execution time in GPU mode: ", time.clock() - GPUstart_time, "seconds"
    
        # Get the activations of the eight layers
        # data[0] indicates first image in the batch
        #print "TEST 11"         
        conv1_output = net.blobs['conv1'].data[0].flatten()
        #print conv1_output.shape
        #print conv1_output
        #print "within###########################################"

        conv2_output = net.blobs['conv2'].data[0].flatten()
        conv3_output = net.blobs['conv3'].data[0].flatten()
        conv4_output = net.blobs['conv4'].data[0].flatten()
        conv5_output = net.blobs['conv5'].data[0].flatten()
        fc6_output = net.blobs['fc6'].data[0].flatten()
        fc7_output = net.blobs['fc7'].data[0].flatten()
        fc8_output = net.blobs['fc8'].data[0].flatten()
    
        # add the layer activations to their respective arrays
        if label == "left":
            lconv1_activations[:,i] = conv1_output 
            lconv2_activations[:,i] = conv2_output
            lconv3_activations[:,i] = conv3_output
            lconv4_activations[:,i]  = conv4_output
            lconv5_activations[:,i] = conv5_output
            lfc6_activations[:,i]  = fc6_output
            lfc7_activations[:,i]  = fc7_output
            lfc8_activations[:,i]= fc8_output

        elif label == "right":
            rconv1_activations[:,i]  =  conv1_output
            rconv2_activations[:,i]  = conv2_output
            rconv3_activations[:,i] = conv3_output
            rconv4_activations[:,i]  = conv4_output
            rconv5_activations[:,i] = conv5_output
            rfc6_activations[:,i] = fc6_output
            rfc7_activations[:,i] = fc7_output
            rfc8_activations[:,i] = fc8_output
  
    
        #if i==0:
        #    break
        
        i += 1   

    #end of method
    #return activation_array

# Paths to directories containing left and right parts
# of the panaromic scene
left_dir = caffe_root + 'examples/panaromic_db/panleft_db/' 
right_dir = caffe_root + 'examples/panaromic_db/panright_db/' 
full_dir = caffe_root + 'examples/panaromic_db/'

# arrays to store avg activations for eight layers
# for each category
#num_categories = 1
#images_percategory = 5

# define the category sequence
#category_seq:  ['creek', 'dessert', 'forest', 'farm', 'field', 'football_field', 'garden', 'dock']
#category_seq =  ['cree', 'dess', 'fore', 'farm', 'fiel', 'foot', 'gard', 'dock']

# k, (i, j)  

#right_activations = np.zeros((num_layers, num_images))
#full_activations = np.zeros((num_layers, num_images))

get_activations(full_dir, "full")
get_activations(left_dir, "left")
get_activations(right_dir, "right")
#print "after ######################"
#print lconv3_activations[60000, 2]
#print lconv3_activations[5000, 5]
#print lconv3_activations[2,40]
#print lconv3_activations[62000, 40]


# store the data in the pickle file
data = {
        'lconv1_activations': lconv1_activations,
        'lconv2_activations': lconv2_activations,
        'lconv3_activations': lconv3_activations,
        'lconv4_activations': lconv4_activations,
        'lconv5_activations': lconv5_activations,
        'lfc6_activations': lfc6_activations,
        'lfc7_activations': lfc7_activations,
        'lfc8_activations': lfc8_activations,

        'rconv1_activations': rconv1_activations,
        'rconv2_activations': rconv2_activations,
        'rconv3_activations': rconv3_activations,
        'rconv4_activations': rconv4_activations,
        'rconv5_activations': rconv5_activations,
        'rfc6_activations': rfc6_activations,
        'rfc7_activations': rfc7_activations,
        'rfc8_activations': rfc8_activations,



        }

pickle.dump(data, open(fname, 'wb'))
print ("pickle complete")
print (fname)
