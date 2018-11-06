from sklearn.svm import SVC
import MNIST
import random


if __name__ == '__main__':

    im = MNIST.MNIST2vector('train-images.idx3-ubyte')
    label = MNIST.decode_idx1_ubyte('train-labels.idx1-ubyte')

    test = MNIST.MNIST2vector('t10k-images.idx3-ubyte')
    test_label = MNIST.decode_idx1_ubyte('t10k-labels.idx1-ubyte')

    train_idx = list(range(10000))
    random.shuffle(train_idx)
    im_sample = im[train_idx]
    label_sample = label[train_idx]

    test_idx = list(range(200))
    random.shuffle(test_idx)
    test_sample = test[test_idx]
    test_label_sample = test_label[test_idx]

    clf = SVC(kernel = 'poly')
    clf.fit(im_sample, label_sample)

    score = clf.score(test_sample, test_label_sample)
    print(" score: {:.6f}".format(score))
