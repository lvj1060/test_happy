# from __future__ import print_function #__future__模块，把下一个新版本的特性导入到当前版本，于是我们就可以在当前版本中测试一些新版本的特性
# 我的Python版本是3.6.4.所以不需要这个

from time import time  # 对程序运行时间计时用的
import logging  # 打印程序进展日志用的
import matplotlib.pyplot as plt  # 绘图用的

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)  # 名人的人脸数据集、

n_samples, h, w = lfw_people.images.shape  # 多少个实例，h,w高度，宽度值

X = lfw_people.data  # 特征向量矩阵
n_feature = X.shape[1]  # 每个人有多少个特征值

Y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]  # 多少类
print("Total dataset size")
print("n_samples:", n_samples)
print("n_feature:", n_feature)
print("n_classes:", n_classes)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)  # 选取0.25的测试集

# 降维
n_components = 150  # PCA算法中所要保留的主成分个数n，也即保留下来的特征个数n
print("Extracting the top %d eigenfaces from %d faces" % (n_components, X_train.shape[0]))
t0 = time()
pca = PCA(svd_solver='randomized', n_components=n_components, whiten=True).fit(X_train)  # 训练一个pca模型

print("Train PCA in %0.3fs" % (time() - t0))

eigenfaces = pca.components_.reshape((n_components, h, w))  # 提取出来特征值之后的矩阵

print("Prijecting the input data on the eigenfaces orthonarmal basis")
t0 = time()
X_train_pca = pca.transform(X_train)  # 将训练集与测试集降维
X_test_pca = pca.transform(X_test)
print("Done PCA in %0.3fs" % (time() - t0))

# 终于到SVM训练了
print("Fiting the classifier to the training set")
t0 = time()
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],  # C是对错误的惩罚
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }  # gamma核函数里多少个特征点会被使用}#对参数尝试不同的值
clf = GridSearchCV(SVC(kernel='rbf'), param_grid)
clf = clf.fit(X_train_pca, Y_train)
print("Done Fiting in %0.3fs" % (time() - t0))

print("Best estimotor found by grid search:")
print(clf.best_estimator_)

print("Predicting people's names on the test set")
t0 = time()
Y_pred = clf.predict(X_test_pca)
print("done Predicting in %0.3fs" % (time() - t0))

print(classification_report(Y_test, Y_pred, target_names=target_names))  # 生成一个小报告呀
print(confusion_matrix(Y_test, Y_pred, labels=range(n_classes)))  # 这个也是，生成的矩阵的意思是有多少个被分为此类。


# 把分类完的图画出来12个。

# 这个函数就是画图的
def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())


# 这个函数是生成一个固定格式的字符串的
def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return "predicted: %s\n true: %s" % (pred_name, true_name)


predicted_titles = [title(Y_pred, Y_test, target_names, i) for i in range(Y_pred.shape[0])]  # 这个for循环的用法很简介

plot_gallery(X_test, predicted_titles, h, w)

eigenfaces_titles = ["eigenface %d " % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenfaces_titles, h, w)

plt.show()