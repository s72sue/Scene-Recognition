import cPickle as pickle
import matplotlib.pyplot as plt
import numpy as np
import sys

if len(sys.argv) > 1:
    fname = sys.argv[1]
else:
    print "Require a file containing data to generate the plot"
    exit(0)

def plot_heat_map(X, heat_array):
    """Plot which parts of an image are particularly import for the
    net to classify the image correctly.
    
    See paper: Zeiler, Fergus 2013
    Parameters
    ----------
    X : numpy.array
      The input data, should be of shape (b, c, h, w). Only makes
      sense with image data.
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
                         
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
        
    ax = axes
    img = X.mean()
    ax[0].imshow(-img, interpolation='nearest', cmap='gray')
    ax[0].set_title('image')
    ax[1].imshow(-heat_array, interpolation='nearest', cmap='Reds')
    ax[1].set_title('critical parts')
    ax[2].imshow(-img, interpolation='nearest', cmap='gray')
    ax[2].imshow(-heat_array, interpolation='nearest', cmap='Reds', alpha=0.6)
    ax[2].set_title('super-imposed') 
    
    return plt      


data = pickle.load(open(fname, 'rb')) 
heat_array = data['heat_array']
transformed_image = data['transformed_image']

plt = plot_heat_map(transformed_image, heat_array)
plt.show()
