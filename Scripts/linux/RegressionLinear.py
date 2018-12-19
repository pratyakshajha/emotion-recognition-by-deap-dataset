import numpy
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


file_x = 'data/features_sampled.dat'
file_y = 'data/labels_0.dat'

X = numpy.genfromtxt(file_x, delimiter=' ')
y = numpy.genfromtxt(file_y, delimiter=' ')
		
# Split the data into training/testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Linear regression object
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
y_predict = regr.predict(X_test)
print(mean_squared_error(y_predict, y_test))