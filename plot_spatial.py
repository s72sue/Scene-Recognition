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
    
# get corridor db results
corr_categories = data['corr_categories']
corr_activations = data['corr_activations']
corr_prob = data['corr_prob']    
cf_prob = data['cf_prob']    
top5_corr = data['top5_corr']  
corr_activations_matrix = data['corr_activations_matrix']

# get forest db results
forest_categories = data['forest_categories']
forest_activations = data['forest_activations']
forest_prob = data['forest_prob']    
fc_prob = data['fc_prob']    
top5_forest = data['top5_forest']    
forest_activations_matrix = data['forest_activations_matrix']

# get conference room db results
conf_categories = data['conf_categories']
conf_activations = data['conf_activations']
conf_prob = data['conf_prob']    
cocl_prob = data['cocl_prob']    
top5_conf = data['top5_conf'] 
conf_activations_matrix = data['conf_activations_matrix']

# get class room db results
class_categories = data['class_categories']
class_activations = data['class_activations']
class_prob = data['class_prob']    
clco_prob = data['clco_prob']    
top5_class = data['top5_class'] 
class_activations_matrix = data['class_activations_matrix']

# plot correlation between Corridor and forest
for_corr = np.concatenate((forest_activations_matrix, corr_activations_matrix), axis=0)

# plot correlation between Classroom and Conference room
class_conf = np.concatenate((class_activations_matrix, conf_activations_matrix), axis=0)

"""
corr = corrcoef(left_right)
x_values = range(1,9,1)
y_values = []
for i in range(len(x_values)):
    y_values.append(corr[i, i+8]) 

plt.figure(facecolor="white")
plt.plot(x_values, y_values)
plt.show()
"""

plot_correlation(for_corr, "Forest scenes and Corridor image")
plot_correlation(class_conf, "Classroom scenes and Conference room scenes")


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


