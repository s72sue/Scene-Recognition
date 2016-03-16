# setup the environment
import numpy as np
import matplotlib.pyplot as plt
import time

# set the defaults for display
plt.rcParams['figure.figsize'] = (10,10)   # image size
plt.rcParams['image.interpolation'] = 'nearest' # show square pixels, don't interpolate
plt.rcParams['image.cmap'] = 'gray' # use grayscale output rather than a colour heat map (which can be misleading)


# Load Caffe
import caffe

# Load the net

# set caffe to cpu/gpu mode
caffe.set_mode_cpu()

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


## CPU classification
# set the size of the input if different from the default
net.blobs['data'].reshape(50,  # batch size
                           3,  # 3-channel BGR images
                           227, 227)  # image size is 227x227

# load an image and perform the preprocessing
#image = caffe.io.load_image(caffe_root + 'examples/images/cat.jpg')
image = caffe.io.load_image(caffe_root + 'scene/placesCNN_upgraded/testSet_resize/fffd911a0e4dcb072b417882d6106b9f.jpg')
transformed_image = transformer.preprocess('data', image)
#print image
plt.imshow(image)
plt.show()


# Classify the image
# copy the image data to the memory allocated for the conv net
net.blobs['data'].data[...] = transformed_image

# perform classification
output = net.forward()

# get the output vector for the first image in the batch 
output_prob = output['prob'][0]  
print "Predicted Category is: ", output_prob.argmax()

# Load the labels for the places database
labels_file = caffe_root + 'scene/placesCNN_upgraded/categoryIndex_places205.csv'
labels = np.loadtxt(labels_file, str, delimiter='\t')
print 'output label: ', labels[output_prob.argmax()]

# Looking at the top five predictions from softmax output
top_inds = output_prob.argsort()[::-1][:5 ]  #reverse sort and take five largest items

print 'probabilities and labels:\n', zip(output_prob[top_inds], labels[top_inds])


# Switching to GPU mode to run the conv net
# To figure out how much speedup it provides

# time in the cpu mode
CPUstart_time = time.clock()
net.forward()
print "Execution time in CPU mode: ", time.clock() - CPUstart_time, "seconds"

# time in the GPU mode
caffe.set_device(0)  # pick the first GPU in case of multiple gpus
caffe.set_mode_gpu()
GPUstart_time = time.clock()
net.forward()
print "Execution time in GPU mode: ", time.clock() - GPUstart_time, "seconds"


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
# data is four dimensional

def vis_4data(data):
    """
    Parameters:
    data: An array of shape (n, height, width, 3) or (n, height, width)
    Output:
    A grid of size approx sqrt(n) by sqrt(n) containing each (height, width)
    """

    # Step1: normalize data for display
    n = int(np.ceit(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
               (0, 1), (0, 1))                 # adding some space between filters
               + ((0, 0),) * (data.ndim - 3))  # the last dimension (if there is one) shouldn't be padded
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)
    
    # tiling the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    
    plt.imshow(data); plt.axis('off')


# Visualize the filter of conv1 i.e. the first layer
# the parameters are a list of [weights, biases]
filters = net.params['conv1'][0].data
vis_4data(filters.transpose(0, 2, 3, 1))

# Visualize the output of conv1 (rectified responses of filters, first 36 only)
output = net.blobs['conv1'].data[0, :36]
vis_square(output)

# Visualize the output of pool5 i.e., the fifth layer
output = net.blobs['pool5'].data[0]
vis_square(output)

# Visualize the output of the fully connected rectified layer fc6
# Plt the output values and the histogram of the positive values
output = net.blobs['fc6'].data[0]
plt.subplot(2, 1, 1)
plt.plot(output.flat)
plt.subplot(2, 1, 2)
plt.hist(output.flat[output.flat > 0], bins=100)

# Visualize the final probability output
feat = net.blobs['prob'].data[0]
plt.figure(figsize=(15, 3))
plt.plot(feat.flat)


# Comput the accuracy 
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

