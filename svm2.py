# from sklearn import svm
import numpy as np
import MFCC_test
from sklearn.svm import SVC

# MFCC_test.MFCC('babycry1.wav')
baby1=MFCC_test.MFCC('babycry1.wav')
# MFCC_test.MFCC('babycry2.wav')
baby2=MFCC_test.MFCC('babycry2.wav')
# MFCC_test.MFCC('babycry3.wav')
baby3=MFCC_test.MFCC('babycry3.wav')
# MFCC_test.MFCC('break1.wav')
baby4=MFCC_test.MFCC('babycry4.wav')
baby5=MFCC_test.MFCC('babycry5.wav')
baby6=MFCC_test.MFCC('babycry6.wav')
break1=MFCC_test.MFCC('break1.wav')
# MFCC_test.MFCC('break2.wav')
break2=MFCC_test.MFCC('break2.wav')
# MFCC_test.MFCC('break3.wav')
break3=MFCC_test.MFCC('break3.wav')
break4=MFCC_test.MFCC('break4.wav')


baby1=baby1.reshape(1,-1)
baby2=baby2.reshape(1,-1)
baby3=baby3.reshape(1,-1)
baby4=baby4.reshape(1,-1)
baby5=baby5.reshape(1,-1)
baby6=baby6.reshape(1,-1)
break1=break1.reshape(1,-1)
break2=break2.reshape(1,-1)
break3=break3.reshape(1,-1)
break4=break4.reshape(1,-1)

# X = np.array([baby1,baby2,baby3,break1,break2,break3])
X = np.concatenate((baby1,baby2,baby3,break1,break2,break3),axis=0)
y = (1,1,1,2,2, 2)

print(X.shape)
print(len(y))

clf = SVC()
clf.fit(X, y)
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)

print ("break4:",break4.shape)
print(clf.predict(baby4))

# MFCC_test.MFCC('0b914d51.wav')
# print(mfcc)
