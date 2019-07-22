from keras.datasets import mnist
from autokeras_NAS import ImageClassifier
from matplotlib import pyplot as plt

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.reshape(x_test.shape + (1,))
    # Strange!!
    # nest line actually calls:
    # NetworkModule class from:
    # anaconda3/envs/auto-keras/lib/python3.6/site-packages/autokeras/net_module.py
    # cos code: "from autokeras import ImageClassifier"
    # when you execute it, it will call autokeras from anaconda env's site-packages
    # however the grammar checker will track it from the local package!!!
    # they are different!!!!!
    clf = ImageClassifier(verbose=True, augment=False)
    clf.fit(x_train, y_train, time_limit=30 * 6000)
    clf.final_fit(x_train, y_train, x_test, y_test, retrain=True)
    y = clf.evaluate(x_test, y_test)
    plt.imshow(x_train[0, :, :, 0], cmap=plt.cm.bone)
    print(y * 100)
