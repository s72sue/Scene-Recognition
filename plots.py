import cPickle as pickle
import matplotlib.pyplot as plt
import numpy as np
import sys

if len(sys.argv) > 1:
    fname = sys.argv[1]
else:
    print "Require a file containing data to generate the plot"
    exit(0)


data = pickle.load(open(fname, 'rb')) 
sigma = data['sigma']
output_list = data['output_list']
category_list = data['category_list']

# figure out how many different categories were provided
# to the network as input
num_categories = len( np.unique(category_list))
print "No of categories: ", num_categories

# figure out how many images of each category were provided
mapping = dict((i, category_list.count(i)) for i in category_list)
#print "Number of images of each category (category, num_images): ", mapping
#plt.hist(category_list, np.unique(category_list), histtype='step')
#plt.show()

plt.figure()
plt.plot(sigma, output_list)
#plt.title ("Output probabilities as a function of std-deviation of gaussian")
plt.xlabel("Standard deviation of the gaussian filter")
plt.ylabel("Normalized confidence level of the network")
plt.show()