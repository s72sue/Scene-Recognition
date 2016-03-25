# This script splits the panaromic images into two halves
# along its width.

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


# configure input and output directories
caffe_root = '../'
directory = caffe_root + 'examples/panaromic_db/' 
right_dir = directory + 'panright_db/'
left_dir = directory + 'panleft_db/'

output_list = []  #avg output from all images
sigma = np.arange(0, 13, 0.5)
i = 0

for filename in os.listdir(directory):
    
     # split the filename into "name" and "jpg"
    new_name = filename.split(".")
   
    if (len(new_name) > 1 and new_name[1] == 'jpg'):
    
        img = Image.open(directory + '/' + filename)
        #plt.figure()
        #plt.imshow(img)

        # split the image into two haves (along its width)
        imgwidth, imgheight = img.size
        img_left = img.crop((0, 0, imgwidth/2, imgheight))
        img_right = img.crop((imgwidth-imgwidth/2, 0, imgwidth, imgheight))
     

        # save the two new images in the target directory
        img_left.save(left_dir + new_name[0] + '_left.jpg')
        img_right.save(right_dir + new_name[0] + '_right.jpg')

        """
        plt.figure()
        plt.imshow(img_left)
        plt.figure()
        plt.imshow(img_right)
        plt.show()
        """
       
   


