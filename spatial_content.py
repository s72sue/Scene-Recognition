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


def forward_pass(filename, db_name):
    # load the image
    image = caffe.io.load_image(db_name + filename)

    transformed_image = transformer.preprocess('data', image)
    #plt.figure()
    #plt.imshow(image)
    #plt.show()

    # Classify the image
    # copy the image data to the memory allocated for the conv net
    net.blobs['data'].data[...] = transformed_image

    # perform classification
    output = net.forward()
    return output

def get_activations():
    
    activations = []
    # Get the activations of the eight layers
    # data[0] indicates first image in the batch
    activations.append( np.mean( net.blobs['conv1'].data[0] ) )
    activations.append( np.mean( net.blobs['conv2'].data[0] ) )
    activations.append( np.mean( net.blobs['conv3'].data[0] ) )
    activations.append( np.mean( net.blobs['conv4'].data[0] ) )
    activations.append( np.mean( net.blobs['conv5'].data[0] ) )
    activations.append( np.mean( net.blobs['fc6'].data[0] ) )
    activations.append( np.mean( net.blobs['fc7'].data[0] ) )
    activations.append( np.mean( net.blobs['fc8'].data[0] ) )
    
    return activations
 
def process_db(db_name, label_index, opp_index):
     
    avg_activation = []
    categories = []
    avg_probability = 0
    opp_probability = 0  # opponent's probability predicted by model (avg)
    opt_probability = 0  # an option 3rd category which might be interesting
    top5_output = np.zeros(205)
    i = 0
    # go over the images in the db
    for filename in os.listdir(db_name):
        output = forward_pass(filename, db_name)
        activations = get_activations()
        
        if i==0:
            avg_activation = activations
        else:
            avg_activation = [sum(x) for x in zip(avg_activation, activations)]
        
        # get the output vector for the first image in the batch 
        output_prob = output['prob'][0]  
        top5_output += output_prob
        pred_category = output_prob.argmax()  
        categories.append(pred_category)
        avg_probability += output_prob.item(label_index)
        opp_probability += output_prob.item(opp_index)
         
        i += 1
          
    avg_activation = [x/i for x in avg_activation]  
    avg_probability = avg_probability/i
    opp_probability = opp_probability/i

    top5_output = top5_output/i
    top_inds = top5_output.argsort()[::-1][:5]  #reverse sort and take 5 largest items
    top5_output = zip(top5_output[top_inds], labels[top_inds])

    return avg_activation, categories, avg_probability, opp_probability, top5_output
     
    

# Paths to directories containing left and right parts
# of the panaromic scene
corridor_dir = caffe_root + 'examples/spatial_db/corridors/' 
forest_dir = caffe_root + 'examples/spatial_db/forest_path/' 
conference_dir = caffe_root + 'examples/spatial_db/conference_room/'
classroom_dir = caffe_root + 'examples/spatial_db/classroom/'

# indices of true labels
corridor_idx = 54
forestpath_idx = 78
conferenceroom_idx = 51
classroom_idx = 44
forestroad_idx = 79

# Process the images in each database
corr_activations, corr_categories, corr_prob, cf_prob, top5_corr = process_db(corridor_dir, corridor_idx, forestpath_idx)
forest_activations, forest_categories, forest_prob, fc_prob, top5_forest = process_db(forest_dir, forestpath_idx, corridor_idx)
conf_activations, conf_categories, conf_prob, cocl_prob, top5_conf = process_db(conference_dir, conferenceroom_idx, classroom_idx)
class_activations, class_categories, class_prob, clco_prob, top5_class = process_db(classroom_dir, classroom_idx, conferenceroom_idx)


print "################################# RESULTS #######################################"

print "############################ Corridor db results ############################"
print "Categories predicted: ", corr_categories
print "Activations: ", corr_activations
print "Probability of Corridor: ", corr_prob
print "Probability of forest_path: ", cf_prob
print "Top 5 Probability: ", top5_corr

print "############################ Forest db results ############################"
print "Categories predicted: ", forest_categories
print "Activations: ", forest_activations
print "Probability of forest_path: ", forest_prob
print "Probability of Corridor: ", fc_prob
print "Top 5 Probability: ", top5_forest


print "############################ Conference room db results ############################"
print "Categories predicted: ", conf_categories
print "Activations: ", conf_activations
print "Probability of Conference room: ", conf_prob
print "Probability of Class room: ", cocl_prob
print "Top 5 Probability: ", top5_conf


print "############################ Class room db results ############################"
print "Categories predicted: ", class_categories
print "Activations: ", class_activations
print "Probability of Class room: ", class_prob
print "Probability of Conference room: ", clco_prob
print "Top 5 Probability: ", top5_class



