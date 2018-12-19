import numpy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

def cross_validate3():
    
    # Get Data
    file_x = 'data/features_sampled.dat'
    file_y = 'data/label_class_3.dat'
    
    X = numpy.genfromtxt(file_x, delimiter=' ')
    y = numpy.genfromtxt(file_y, delimiter=' ')
    
    X = StandardScaler().fit_transform(X)
    
    # Split the data into training/testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    	
    models = []
    models.append(('LR', LogisticRegression()))
    models.append(('SVC', SVC()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('DT', DecisionTreeClassifier()))
    
    scoring = 'accuracy'
    
    # Cross Validate
    results = []
    names = []
    timer = []
    print('Model | Mean of CV | Std. Dev. of CV | Time')
    for name, model in models:
        start_time = time.time()
        kfold = model_selection.KFold(n_splits=5, random_state=42)
        cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
        t = (time.time() - start_time)
        timer.append(t)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f) %f s" % (name, cv_results.mean(), cv_results.std(), t)
        print(msg)
        
if __name__ == '__main__':
    cross_validate3()