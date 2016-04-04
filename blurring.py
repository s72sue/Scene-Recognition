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


# Apply gaussian filtering to the image
#image_array = misc.imread(caffe_root + 'examples/blurr_db/bedroom.jpg')
directory = caffe_root + 'scene/placesCNN_upgraded/testSet_resize' 
category_list = []
output_list = []  #avg output from all images
sigma = np.arange(0, 13, 0.5)
i = 0
for filename in os.listdir(directory):
    image_array = np.array(misc.imread(directory + '/' + filename))

    #print type(image_array)
    #print image_array.shape

    # stores the output prob of a given image
    imoutput_list = [] 
    for std_devtn in sigma:
        blurred_image = ndimage.gaussian_filter(image_array, sigma=std_devtn)
        img = Image.fromarray(blurred_image.astype(np.uint8), 'RGB')
        #plt.figure()
        #plt.imshow(img)
        #plt.title("blurred image with std deviation =")
        img_name = caffe_root + 'examples/blurr_db/' +  'blurr' + str(i) + '_' + str(std_devtn) + '.jpg'
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

        # get the output vector for the first image in the batch 
        output_prob = output['prob'][0]  
        highest_prob = max(output_prob)
        pred_category = output_prob.argmax()

        if std_devtn == 0.0:
            category_list.append(pred_category)
            imoutput_list.append(highest_prob)
        else:
            # for computing confidence wrt correct category
            imoutput_list.append(output_prob(category_list[i]))

            
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

fname = "blurring_corrcategory_result.p"
pickle.dump(data, open(fname, 'wb'))
print ("pickle complete")
print (fname)
