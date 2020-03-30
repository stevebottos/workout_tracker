import numpy as np
import cv2
import os
from tensorflow.keras import layers, models, callbacks
from sklearn.utils import class_weight


def makeDatasetInMemory(class_folders,
                        in_path="train/"):
    # Load in the train data
    train_images = []
    train_labels = []
    # one_hot_encoder = np.zeros(len(class_folders))

    for c in class_folders:
        class_label_indexer = int(c[5])-1  # TODO: Make this more robust, will break if double digits
        print("loading class", class_label_indexer)
        for f in os.listdir(in_path + c):
            im = cv2.imread(in_path + c + f, 0)
            im = cv2.resize(im, (220, 220))
            train_images.append(im)

            # Don't think I need to one-hot encode, TF docs say to pass index of class not multi-class
            # vector labels
            # label = np.copy(one_hot_encoder)
            # label[class_label_indexer] = 1
            # train_labels.append(label)

            train_labels.append(class_label_indexer)

    train_images = np.array(train_images)
    train_labels = np.array(train_labels)

    train_images = train_images / 255  # Normalize

    #TODO: Shuffle these two boys together to maintain indices
    return train_images, train_labels

def modelInit():

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(h, w, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(3, activation='softmax'))

    # model.summary()

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def pipeline(dataset):
    dataset = np.array(dataset)
    dataset = dataset / 255  # Normalize
    h, w = dataset[0].shape
    n = len(dataset)
    dataset = dataset.reshape(n, h, w, 1)

    return dataset

### MODES:
train_model = 0
get_predictions = 1
test_model = 0

if train_model:

    class_folders = ["class1/", "class2/", "class3/"]
    train_labels, train_images = makeDatasetInMemory(class_folders)

    # Some slight pre-processing
    train_images = pipeline(train_images)
    class_weights = class_weight.compute_sample_weight('balanced', train_labels)

    model = modelInit()
    model.fit(train_images, train_labels, epochs=4, class_weight = class_weights)
    model.save('cnn_1.h5')

if get_predictions:
    m = models.load_model("cnn_1.h5")

    # Load in some test data
    test_images = []
    test_dir = "test/raw_images/"

    cnt = 1
    # lim = 500
    for f in os.listdir(test_dir):
        print(test_dir+f)
        im = cv2.imread(test_dir + f, 0)
        im = cv2.resize(im, (220, 220))
        test_images.append(im)

        # if cnt == lim:
        #     break
        cnt += 1

    test_images = pipeline(test_images)

    predictions = m.predict(test_images)
    print(predictions)
    # predictions = np.argmax(predictions, 1).T
    np.savetxt('predictions.csv', predictions, delimiter = ',')

if test_model:
    test_images = []

    dirr = "/home/steve/Qimia Inc Dropbox/Steve Bottos/color_top/images"
    labels = np.genfromtxt('predictions.csv')

    pos = 0
    for f in os.listdir(dirr):
        if labels[pos] == 1:
            print(pos)
            im = cv2.imread(dirr + '/' + f)
            im = cv2.resize(im, None, fx=0.2, fy=0.2)
            cv2.imwrite("1_test/" + f, im)
        # else:
        #     print(pos)
        #     im = cv2.imread(dirr + '/' + f)
        #     im = cv2.resize(im, None, fx=0.2, fy=0.2)
        #     cv2.imwrite("0_test/" + f, im)

        pos += 1
