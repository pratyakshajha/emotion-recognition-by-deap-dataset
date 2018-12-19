import numpy
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

def svm_classifier(): 
    file_x = 'data/features_sampled.dat'
    file_y = 'data/label_class_0.dat'
    
    X = numpy.genfromtxt(file_x, delimiter=' ')
    y = numpy.genfromtxt(file_y, delimiter=' ')
    
    # Split the data into training/testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    	
    # SVM Classifier
    clf = SVC()
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_predict)
    print(cm)
    print(accuracy_score(y_test, y_predict))
    
if __name__ == '__main__':
    svm_classifier()