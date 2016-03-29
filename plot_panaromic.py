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


# plotting the correlation matrix
def plot_correlation(matrix, title):
    plt.figure(facecolor='white')
    R = corrcoef(matrix)
    pcolor(R)
    colorbar()
    xticks(arange(0.5, 16.5), range(1,17,1))
    yticks(arange(0.5, 16.5), range(1,17,1))
    plt.title(title)
    show()

        
data = pickle.load(open(fname, 'rb'))
left_activations = data['left_activations']
right_activations = data['right_activations']
full_activations = data['full_activations']


left_right = np.concatenate((left_activations, right_activations), axis=0)
left_full = np.concatenate((left_activations, full_activations), axis=0)
right_full = np.concatenate((right_activations, full_activations), axis=0)


corr = corrcoef(left_right)
x_values = range(1,9,1)
y_values = []
for i in range(len(x_values)):
    y_values.append(corr[i, i+8]) 

plt.figure(facecolor="white")
plt.plot(x_values, y_values)
plt.show()


plot_correlation(left_right, "left segment and right image")
#plot_correlation(left_full, "left segment and full image")
#plot_correlation(right_full, "right segment and full image")


"""
plt.figure()
plt.plot(range(1,9,1), left_activations, 'r--')
plt.plot(range(1,9,1), right_activations, 'g--')



correlation = pearsonr(left_activations, right_activations)
sig_correlation = np.correlate(left_activations, right_activations)
mse = []
for i in range(8):
    mse.append((np.abs(left_activations[i]-right_activations[i])/np.abs(max(left_activations[i], right_activations[i]))) * 100)


print "mse:" , mse
print "Pearson Correlations: ", correlation
print "Signal correlation: ", sig_correlation

plt.figure(facecolor='white')
plt.plot(range(1,9,1), mse)
plt.show()

print "NO OF IMAGES: " , len(right_activations)
"""
