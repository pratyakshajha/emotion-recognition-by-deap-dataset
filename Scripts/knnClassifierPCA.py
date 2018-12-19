import numpy
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

def knn_classifier_pca():
    file_x = 'data/features_raw.dat'
    file_y = 'data/label_class_0.dat'
    
    X = numpy.genfromtxt(file_x, delimiter=' ')
    y = numpy.genfromtxt(file_y, delimiter=' ')
    
    # PCA to select features
    pca = PCA(n_components=2, svd_solver='full')
    pca.fit(X)
    X = pca.transform(X)
    
    # Split the data into training/testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    	
    # Logistic regression
    clf = KNeighborsClassifier()
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_predict)
    print(cm)
    print(accuracy_score(y_test, y_predict))
    
if __name__ == '__main__':
    knn_classifier_pca()