# setup the environment
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy
from scipy import misc
from scipy import ndimage
from PIL import Image
import cPickle as pickle
import os

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


"""
# Code to test the image conversion to an array
img_original = Image.fromarray(image_array, 'RGB')
plt.imshow(img_original)
plt.show()
"""


# Apply gaussian filtering to the image
#image_array = misc.imread(caffe_root + 'examples/blurr_db/bedroom.jpg')
directory = caffe_root + 'scene/placesCNN_upgraded/testSet_resize' 
category_list = []
output_list = []  #avg output from all images
sigma = np.arange(0, 13.5, 0.5)
i = 0
for filename in os.listdir(directory):
    image_array = np.array(misc.imread(directory + '/' + filename))
    im_arr_0 =  image_array[:,:,0]
    im_arr_1 =  image_array[:,:,1]
    im_arr_2 =  image_array[:,:,2]
    #blurred_image = 
    #print np.concatenate((im_arr_0[:,:,None], im_arr_1[:,:,None], im_arr_2[:,:,None]), axis=2).shape
    """
    print image_array
    print image_array.shape
    print image_array[:,:,0].shape
    print image_array[:,:,1].shape
    print image_array[:,:,2].shape
    print image_array[:,:,3].shape
    """
    #print "####################################"

    #print type(image_array)
    #print image_array.shape

    # stores the output prob of a given image
    imoutput_list = [] 
    for std_devtn in sigma:
        #print "INSIDE"
        blur_arr_0 = ndimage.gaussian_filter(im_arr_0, sigma=std_devtn)
        blur_arr_1 = ndimage.gaussian_filter(im_arr_1, sigma=std_devtn)
        blur_arr_2 = ndimage.gaussian_filter(im_arr_2, sigma=std_devtn) 
        blurred_image =  np.concatenate((blur_arr_0[:,:,None], blur_arr_1[:,:,None], blur_arr_2[:,:,None]), axis=2)
        img = Image.fromarray(blurred_image.astype(np.uint8), 'RGB')
        #plt.figure()
        #plt.imshow(img)
        #plt.title("blurred image with std deviation =")
        img_name = caffe_root + 'examples/blurr_db_final/' +  'blurr' + str(i) + '_' + str(std_devtn) + '.jpg'
        misc.imsave(img_name, img)


        # load an image and perform the preprocessing
        image = caffe.io.load_image(img_name)

        transformed_image = transformer.preprocess('data', image)
        #print image
        #plt.figure()
        #plt.imshow(image)
        #plt.show()


        # Classify the image
        # copy the image data to the memory allocated for the conv net
        net.blobs['data'].data[...] = transformed_image

        # perform classification
        output = net.forward()
        #print "Execution time in GPU mode: ", time.clock() - GPUstart_time, "seconds"


        # get the output vector for the first image in the batch 
        output_prob = output['prob'][0] 
        highest_prob = max(output_prob)
        #print "Highest probability is: ", highest_prob 
        pred_category = output_prob.argmax()
        #print "Predicted Category is: ", pred_category
        
        if std_devtn == 0.0:
            category_list.append(pred_category)
            imoutput_list.append(highest_prob)
        else:
            # for computing confidence wrt correct category
            imoutput_list.append(output_prob[category_list[i]])
            
            # for computing set to zero confidence
            """
            if pred_category == category_list[i]:
                # append to the list of probabilities
                imoutput_list.append(highest_prob)
            else:
                # the network predicted the wrong answer
                #print "category mismatch"
                imoutput_list.append(0)
            """
            
    # normalize the probabilities such that they lie from (0,1)
    imoutput_list = np.divide(imoutput_list, max(imoutput_list))
    # elementwise addition of lists
    if i == 0:
        output_list = imoutput_list
    else:    
        output_list = [sum(x) for x in zip(output_list, imoutput_list)]
    # breaking condition
    if i == 50:
        break
    i += 1  #increment the index for the next iteration
 

# divide the total by i+1 to compute the average
output_list = np.divide(output_list, i+1)

plt.figure()
plt.plot(sigma, output_list)
plt.title("Plot of output probability as a function of std deviation")
plt.show()

# store the data in a pickle file for easy plot reconstruction
# this helps to view the plot without having to run the script again
data = {
        'sigma': sigma,
        'output_list': output_list,
        'category_list': category_list
        }

fname = "blurring_corrcategory_result_final.p"
pickle.dump(data, open(fname, 'wb'))
print ("pickle complete")
print (fname)

"""
# Load the labels for the places database
labels_file = caffe_root + 'scene/placesCNN_upgraded/categoryIndex_places205.csv'
labels = np.loadtxt(labels_file, str, delimiter='\t')
print 'output label: ', labels[output_prob.argmax()]

# Looking at the top five predictions from softmax output
top_inds = output_prob.argsort()[::-1][:5 ]  #reverse sort and take five largest items
print 'probabilities and labels:\n', zip(output_prob[top_inds], labels[top_inds])


## Look at some of the parameters and intermediate activations
# Obtain the activation shapes for each layer. They should have the
# form (batch_size, channel_dim, height, width).

# look at the output shape for each layer
print "Activation Shapes"
for layer_name, blob in net.blobs.iteritems():
    print layer_name + "  " + str(blob.data.shape)


# look at the parameter shapes. They should have the form
# (output_channels, input_channels, filter_height, filter_width) 
# for the weights and 1-dimensional shape (output_channels) for the biases.
print "Parameter Shapes"
for layer_name, param in net.params.iteritems():
    print layer_name + '\t' + str(param[0].data.shape), str(param[1].data.shape)



## Visualize sets of rectangular heat maps
# take an array of shape (n, height, width) or (n, height, width, channel)
# and Visualize each (height, width) in a grid of size approx sqrt(n) x sqrt(n).
# data is four dimensional

def vis_4data_original(data):

    #Parameters:
    #data: An array of shape (n, height, width, 3) or (n, height, width)
    #Output:
    #A grid of size approx sqrt(n) by sqrt(n) containing each (height, width)


    # Step1: normalize data for display
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
        (0, 1), (0, 1))                 # adding some space between filters
               + ((0, 0),) * (data.ndim - 3))  # the last dimension (if there is one) shouldn't be padded
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)
    
    # tiling the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
     
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    plt.imshow(data); plt.axis('off')
    plt.show()
    

def vis_4data (data, padsize=1, padval=0, title="Image"):
    
    #Parameters:
    #data: An array of shape (n, height, width, 3) or (n, height, width)
    #padsize: size of padding to be used
    #padval: values to use for padding
    #Output:
    #A grid of size approx sqrt(n) by sqrt(n) containing each (height, width)
    
    data -= data.min()
    data /= data.max()

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    plt.figure()
    plt.imshow(data)
    plt.axis('off')
    plt.title(title)
    #plt.show()


# Visualize the filter of conv1 i.e. the first layer
# the parameters are a list of [weights, biases]
# CONV1 FILTERS
filters = net.params['conv1'][0].data
vis_4data(filters.transpose(0, 2, 3, 1), title="CONV1 FILTERS")

# Visualize the output of conv1 (rectified responses of filters, first 36 only)
# CONV1 OUTPUT
output = net.blobs['conv1'].data[0, :36]
vis_4data(output, padval=1, title="CONV1 OUPUT")

# CONV2 FILTERS (FIRST 48)
filters = net.params['conv2'][0].data
vis_4data(filters[6].reshape(48*1, 5, 5), title="CONV2 FILTERS - each channel is shown separetely so that each filter is a row")


# CONV2 OUTPUT
output = net.blobs['conv2'].data[0, :36]
vis_4data(output, padval=1, title="CONV2 OUTPUT")

# CONV3 OUPUT
output = net.blobs['conv3'].data[0]
vis_4data(output, padval=0.5, title="CONV3 OUTPUT - all 384 channels")

plt.show()

# CONV4 OUTPUT
output = net.blobs['conv4'].data[0]
vis_4data(output, padval=0.5, title="CONV4 OUTPUT - all 384 channels")

# CONV5 OUTPUT
output = net.blobs['conv5'].data[0]
vis_4data(output, padval=0.5, title="CONV5 OUTPUT - all 256 channels")


#Filters of conv5
filters = net.params['conv5'][0].data[0, 50:114]
vis_4data(filters, padval=1, title="FILTERS OF CONV5 - index 50 - 114")


# Visualize the output of conv5 (rectified responses of filters, first 36 only)
output = net.blobs['conv5'].data[0, :36]
vis_4data(output, padval=1, title="CONV5 OUTPUT = first 36 channels only")


# POOL5 OUTPUT
# Visualize the output of pool5 i.e., the fifth layer
output = net.blobs['pool5'].data[0]
vis_4data(output, padval=1, title="POOL5 OUTPUT")

# Visualize the output of the fully connected rectified layer fc6
# Plt the output values and the histogram of the positive values
output = net.blobs['fc6'].data[0]
plt.subplot(2, 1, 1)
plt.plot(output.flat)
plt.subplot(2, 1, 2)
plt.hist(output.flat[output.flat > 0], bins=100)
plt.title("Output of fc6")
plt.show()

# Visualize the final probability output
feat = net.blobs['prob'].data[0]
plt.figure(figsize=(15, 3))
plt.plot(feat.flat)
plt.title("Final probability output")
plt.show()

v = np.zeros((13, 13))

m = 0
for x in xrange(0, 13):
    for y in xrange(0, 13):
        s = 0
        for filter_n in xrange(0, 255):
            s += net.blobs['conv5'].data[0][filter_n][x][y]
        m = max(s, m)
        v[x][y] = s

for x in xrange(0, 13):
    for y in xrange(0, 13):
        v[x][y] /= m

plt.figure()
plt.imshow(v)
plt.show()

plt.figure()
plt.imshow(net.blobs['conv5'].data[0][240])
plt.show()

plt.figure()
plt.imshow(net.blobs['pool5'].data[0][240])
plt.show()


# Compute the accuracy 
accuracy = 0
num_iterations = 10
for i in range(num_iterations):
    # one iteration, images=batch size
    outputs = net.forward()
    accuracy += net.blobs['accuracy-top5'].data
    print outputs
print ('############################################################') 
avg_accuracy = accuracy/num_iterations
print "Accuracy = ", avg_accuracy

"""

