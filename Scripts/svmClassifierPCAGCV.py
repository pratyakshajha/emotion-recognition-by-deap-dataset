import numpy
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import RandomizedSearchCV
 
def svm_classifier_pca_gcv():
    file_x = 'data/features_raw.dat'
    file_y = 'data/label_class_0.dat'
    
    X = numpy.genfromtxt(file_x, delimiter=' ')
    y = numpy.genfromtxt(file_y, delimiter=' ')
    
    # PCA to select features
    pca = PCA(n_components=20, svd_solver='full')
    pca.fit(X)
    X = pca.transform(X)
    
    
    # Split the data into training/testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    
    # Tune algorithm
    pipeline = make_pipeline(SVC(random_state=42))
    
    hyperparameters = { 'svc__kernel' : ['poly', 'rbf','sigmoid'],
                      'svc__tol': [1e-3, 1e-1]}
    
    # SVM Classifier
    print('Started RandomizedSearchCV')
    clf = RandomizedSearchCV(pipeline, hyperparameters, cv=10, scoring = 'accuracy')
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    print("MSE for RandomizedSearchCV: {}". format(accuracy_score(y_test, y_predict)))
    cm = confusion_matrix(y_test, y_predict)
    print(cm)

if __name__ == '__main__':
    svm_classifier_pca_gcv()