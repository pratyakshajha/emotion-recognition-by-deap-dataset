from ConvertData import convertData
from FeaturesSampled import sampleFeatures
from LabelClass0 import onehotencoding0
from LabelClass1 import onehotencoding1
from LabelClass2 import onehotencoding2
from LabelClass3 import onehotencoding3
from cross_validation0 import cross_validate0
from cross_validation1 import cross_validate1
from cross_validation2 import cross_validate2
from cross_validation3 import cross_validate3

if __name__ == '__main__':
    print('Load Data:\n')
    convertData()
    print('Downsample Data:\n')
    sampleFeatures()
    print('Encoding Classes:\n')
    onehotencoding0()
    onehotencoding1()
    onehotencoding2()
    onehotencoding3()
    print('Begin Cross Validation:\n')
    print('\nLabel 0\n')
    cross_validate0()
    print('\nLabel 1\n')
    cross_validate1()
    print('\nLabel 2\n')
    cross_validate2()
    print('\nLabel 3\n')
    cross_validate3()