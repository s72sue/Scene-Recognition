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


figure_seq = []
def get_activations(dir_name, activation_array, label):
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
    
        transformed_image = transformer.preprocess('data', image)
        #plt.figure()
        #plt.imshow(image)
        #plt.show()
    
        # Classify the image
        # copy the image data to the memory allocated for the conv net
        net.blobs['data'].data[...] = transformed_image
    
        # perform classification
        output = net.forward()
        #print "Execution time in GPU mode: ", time.clock() - GPUstart_time, "seconds"
    
        # Get the activations of the eight layers
        # data[0] indicates first image in the batch
        conv1_output = net.blobs['conv1'].data[0]
        conv2_output = net.blobs['conv2'].data[0]
        conv3_output = net.blobs['conv3'].data[0]
        conv4_output = net.blobs['conv4'].data[0]
        conv5_output = net.blobs['conv5'].data[0]
        fc6_output = net.blobs['fc6'].data[0]
        fc7_output = net.blobs['fc7'].data[0]
        fc8_output = net.blobs['fc8'].data[0]
    
        # add the avg activations to the list
        activation_array[0, i] = np.mean(conv1_output)
        activation_array[1, 1] = np.mean(conv2_output)
        activation_array[2, i] = np.mean(conv3_output)
        activation_array[3, i] = np.mean(conv4_output)
        activation_array[4, i] = np.mean(conv5_output)
        activation_array[5, i] = np.mean(fc6_output)
        activation_array[6, i] = np.mean(fc7_output)
        activation_array[7, i] = np.mean(fc8_output)
    
        #if i==0:
        #    break
        
        i += 1   

    return activation_array

# Paths to directories containing left and right parts
# of the panaromic scene
left_dir = caffe_root + 'examples/panaromic_db/panleft_db/' 
right_dir = caffe_root + 'examples/panaromic_db/panright_db/' 
full_dir = caffe_root + 'examples/panaromic_db/'

# arrays to store avg activations for eight layers
# for each category
#num_categories = 1
#images_percategory = 5
num_images = 100
num_layers = 8
# define the category sequence
#category_seq:  ['creek', 'dessert', 'forest', 'farm', 'field', 'football_field', 'garden', 'dock']
#category_seq =  ['cree', 'dess', 'fore', 'farm', 'fiel', 'foot', 'gard', 'dock']

# k, (i, j)  
left_activations = np.zeros((num_layers, num_images))
right_activations = np.zeros((num_layers, num_images))
full_activations = np.zeros((num_layers, num_images))

full_activations = get_activations(full_dir, full_activations, "full")
left_activations = get_activations(left_dir, left_activations, "left")
right_activations = get_activations(right_dir, right_activations, "right")

# store the data in the pickle file
data = {
        'left_activations': left_activations,
        'right_activations': right_activations,
        'full_activations':full_activations,
        }

pickle.dump(data, open(fname, 'wb'))
print ("pickle completer")
print (fname)
