from keras.datasets import mnist
from autokeras import ImageClassifier
from matplotlib import pyplot as plt

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.reshape(x_test.shape + (1,))
    clf = ImageClassifier(verbose=True, augment=False)
    clf.fit(x_train, y_train, time_limit=30 * 6000)
    clf.final_fit(x_train, y_train, x_test, y_test, retrain=True)
    y = clf.evaluate(x_test, y_test)
    plt.imshow(x_train[0, :, :, 0], cmap=plt.cm.bone)
    print(y * 100)
