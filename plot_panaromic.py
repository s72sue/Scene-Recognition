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
left_activations = data['left_activations']
right_activations = data['right_activations']
full_activations = data['full_activations']

plt.figure()
plt.plot(range(1,9,1), left_activations, 'r--')
plt.plot(range(1,9,1), right_activations, 'g--')


correlation = pearsonr(left_activations, right_activations)
sig_correlation = np.correlate(left_activations, right_activations)
mse = []
for i in range(8):
    mse.append((np.abs(left_activations[i]-right_activations[i])/np.abs(max(left_activations[i], right_activations[i]))) * 100)


print "mse:" , mse
print "Correlations: ", correlation
print "Signal correlation: ", sig_correlation

plt.figure()
plt.plot(range(1,9,1), mse)
plt.show()
