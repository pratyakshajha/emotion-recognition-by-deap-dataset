from ConvertData import convertData
from FeaturesSampled import sampleFeatures
from LabelClass0 import onehotencoding0
from svmClassifier import svm_classifier
from svmClassifierPCA import svm_classifier_pca
from svmClassifierPCAGCV import svm_classifier_pca_gcv
from knnClassifier import knn_classifier
from knnClassifierPCA import knn_classifier_pca
from etClassifierPCAGCV import etClassifier_pca_gcv
from LDA import lda_classifier

if __name__ == '__main__':
    print('Load Data:\n')
    convertData()
    print('Downsampling Original Data:\n')
    sampleFeatures()
    print('Encoding Classes:\n')
    onehotencoding0()
    print('Simple SVM classification: \n')
    svm_classifier()
    print('\nSVM classification after PCA: \n')
    svm_classifier_pca()
#    print('SVM classification with PCA and GCV: \n\n')
#    svm_classifier_pca_gcv()
#    print('Simple KNN classification: \n\n')
#    knn_classifier()
#    print('PCA +  KNN classification: \n\n')
#    knn_classifier_pca()
#    print('Extra Tree classification with PCA and GCV: \n\n')
#    etClassifier_pca_gcv()
#    print('LDA based classification: \n\n')
#    lda_classifier()