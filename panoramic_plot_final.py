import cPickle as pickle
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.stats.stats import pearsonr
from pylab import pcolor, show, colorbar, xticks, yticks
from numpy import corrcoef, arange

if len(sys.argv) > 1:
    fname = sys.argv[1]
else:
    print "Require a file containing data to generate the plot"
    exit(0)


# method for plotting the correlation matrix
def plot_correlation(matrix, title):
    plt.figure(facecolor='white')
    R = corrcoef(matrix)
    pcolor(R)
    colorbar()
    xticks(arange(0.5, 16.5), range(1,17,1))
    yticks(arange(0.5, 16.5), range(1,17,1))
    plt.title(title, fontsize=16)
    show()


## Read data from the pickle file        
data = pickle.load(open(fname, 'rb'))

lconv1_activations = data[lconv1_activations]
lconv2_activations = data[lconv2_activations]
lconv3_activations = data[lconv3_activations]
lconv4_activations = data[lconv4_activations]
lconv5_activations = data[lconv5_activations]
lfc6_activations = data[lfc6_activations]
lfc7_activations = data[lfc7_activations]
lfc8_activations = data[lfc8_activations]

rconv1_activations = data[rconv1_activations]
rconv2_activations = data[rconv2_activations]
rconv3_activations = data[rconv3_activations]
rconv4_activations = data[rconv4_activations]
rconv5_activations = data[rconv5_activations]
rfc6_activations = data[rfc6_activations]
rfc7_activations = data[rfc7_activations]
rfc8_activations = data[rfc8_activations]


left_right = np.concatenate((left_activations, right_activations), axis=0)
left_full = np.concatenate((left_activations, full_activations), axis=0)
right_full = np.concatenate((right_activations, full_activations), axis=0)

# plotting the correlation coefficient plots
corr_lr = corrcoef(left_right)
corr_rf = corrcoef(right_full)
corr_lf = corrcoef(left_full)

x_values = range(1,9,1)
y_values = []
for i in range(len(x_values)):
    y_values.append(corr_lr[i, i+8]) 

plt.figure(facecolor="white")
plt.plot(x_values, y_values)
plt.xlabel("Layer number", fontsize=16)
plt.ylabel("Correlation coefficient", fontsize=16)
plt.title("left segment and right segment", fontsize=16)
plt.show()

# plotting the correlation matrices
plot_correlation(left_right, "left and right segments")
plot_correlation(left_full, "left segment and full image")
plot_correlation(right_full, "right segment and full image")


plt.figure()
plt.plot(range(1,9,1), left_activations, 'r--')
plt.plot(range(1,9,1), right_activations, 'g--')

# print the total number of images
print "NO OF IMAGES: " , len(right_activations)
