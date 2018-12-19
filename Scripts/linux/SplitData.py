import numpy
from sklearn.model_selection import train_test_split

file_x = 'data/features_raw.dat'
file_y = 'data/labels_0.dat'
file_y_test = 'data/labels_test_0.dat'

X = numpy.genfromtxt(file_x, delimiter=' ')
y = numpy.genfromtxt(file_y, delimiter=' ')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
numpy.savetxt(file_y_test, y_test)
numpy.savetxt("data/features_train.dat",X_train)
numpy.savetxt("data/features_test.dat",X_test)