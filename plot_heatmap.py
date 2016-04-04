import cPickle as pickle
import matplotlib.pyplot as plt
import numpy as np
import sys

if len(sys.argv) > 1:
    fname = sys.argv[1]
else:
    print "Require a file containing data to generate the plot"
    exit(0)

def plot_heat_map(X, heat_array, square_length):
    """Plot which parts of an image are important for the
    net to classify the image correctly.
    
    Parameters
    ----------
    X : numpy.array
      The input data, should be of shape (K, H, W). 
    figsize : tuple (int, int)
      Size of the figure.
    heat_array : np.array (with same size as image)
      An 2D np.array that at each point (i, j) contains the predicted
      probability of the correct class if the image is occluded by a
      square with center (i, j).
    
    Output: Plots
    -----
    Figure with 3 subplots: the original image, the occlusion heatmap,
    and both images super-imposed.
    """

    if X.ndim != 3:
        raise ValueError("This function requires the input data to be of "
                         "shape (K, H, W), instead got {}".format(x.shape))
    c, h, w = X.shape   
    figs, axes = plt.subplots(1, 3, figsize=(w,1*w/3), facecolor='white')   
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
        
    ax = axes
    img = X.mean(0)
    ax[0].imshow(-img, interpolation='nearest', cmap='gray')
    ax[0].set_title('image')
    ax[1].imshow(-heat_array, interpolation='nearest', cmap='Reds')
    ax[1].set_title(('critical parts with square length: ', square_length))
    ax[2].imshow(-img, interpolation='nearest', cmap='gray')
    ax[2].imshow(-heat_array, interpolation='nearest', cmap='Reds', alpha=0.6)
    ax[2].set_title('super-imposed') 
    
    return plt  

# load the heatmap data computed from occlusion.py 
# for a singel square length (heat_array) or from 
# occlusion_batch.py for multiple square lengths (heat_arrdict)
data = pickle.load(open(fname, 'rb'))
transformed_image = data['transformed_image']

length_list = []  #square length list
heat_avg = []

# This plots the effect of area of occluded region
# on the confidence level of the model. Avg confidence level
# is computed by taking the mean of each heatmap generated
# with a particulat square length. 
if 'heat_arrdict' in data:
    heat_arrdict = data['heat_arrdict']
    for key, value in heat_arrdict.iteritems():
        heat_array = value
        length_list.append(key**2)
        heat_avg.append(np.mean(heat_array)*100)
        
        #plt = plot_heat_map(transformed_image, heat_array, key)
        #plt.show() 
    
    # plot the     
    plt.figure(facecolor='white')
    plt.plot(length_list, heat_avg, 'ro')
    plt.show()
else:   
    # this just plots one heatmap for a single square length
    heat_array = data['heat_array']
    plt = plot_heat_map(transformed_image, heat_array)
    plt.show()
