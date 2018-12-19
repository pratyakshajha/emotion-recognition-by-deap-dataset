import pandas as pd
import seaborn as sns
import numpy
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

features = 'data/features_sampled.dat'
labels = 'data/label_class_0.dat'

features = numpy.genfromtxt(features, delimiter=' ')
features = pd.DataFrame(data = features, dtype = numpy.int8)
labels = numpy.genfromtxt(labels, delimiter=' ')
labels = pd.DataFrame(data = labels, dtype = numpy.int8)

names = ['valence', 'arousal', 'dominance', 'liking']
dataset = pd.concat([labels, features], axis=1, join='outer').reindex()


# Summary of dataset
print('Summary of dataset\n')
# shape
print('1. Shape is:')
print(dataset.shape)
# head
print('\n2. First 8 rows are as:')
print(dataset.head(8))
# descriptions
print('\n3. Statistical description:')
print(dataset.describe())
types = dataset.dtypes
print('\n4. Data Types:')
print(types)

# Data visualisation
# box and whisker plots
for i in range(40):
    sns.set_style('whitegrid')
    sns.boxplot(data=dataset[i])
    plt.title('Magnitude versus ' + str(i+1) + ' channel')
    plt.show()
    
for i in range(4):
    sns.set_style('whitegrid')
    sns.violinplot(data=dataset[(str(names[i]))])
    plt.title('Magnitude versus ' + names[i])
    plt.show()