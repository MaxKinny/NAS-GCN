from autokeras import ImageClassifier
from matplotlib import pyplot as plt
import glob
from PIL import Image
import numpy as np

if __name__ == '__main__':
    picture_path = '../../../kaggle_SIIM-ACR-Pneumothorax-Segmentation/data/png/256/train/picture/*.png'
    mask_path = '../../../kaggle_SIIM-ACR-Pneumothorax-Segmentation/data/png/256/train/mask/*.png'
    train_files = glob.glob(picture_path)
    mask_files = glob.glob(mask_path)
    x_train = np.array([np.array(Image.open(fn)) for fn in train_files])
    y_train = []
    for fp in mask_files:
        if len(np.unique(Image.open(fp))) == 2:
            y_train.append(1)
        else:
            y_train.append(0)
    y_train = np.array(y_train)
    clf = ImageClassifier(verbose=True, augment=False)
    clf.fit(x_train, y_train, time_limit=30 * 6000)
    clf.final_fit(x_train, y_train, x_test, y_test, retrain=True)
    y = clf.evaluate(x_test, y_test)
    plt.imshow(x_train[0, :, :, 0], cmap=plt.cm.bone)
    print(y * 100)
