import numpy as np
import tensorflow as tf
import PIL as pil
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MultiLabelBinarizer
from collections import Counter
import pandas

from sklearn.metrics import r2_score

batchsize = 64
image_width = 200
image_height = 200

classes = ['healthy', 'complex', 'scab', 'rust', 'powdery_mildew', 'frog_eye_leaf_spot']

def load_resize_images(file, n0, n):
    file_list = os.listdir(file)
    #making a subset of first 20 images
    file_list = file_list[n0:n]
    images = []
    for img in file_list:
        tempImage = pil.Image.open(os.path.join(file, img))
        imageBox = tempImage.getbbox()
        cropped = tempImage.crop(imageBox)
        # splits the image into separate RGB channels
        r, g, b = tempImage.split()
        # creates a new image with only the red channel
        redscale = pil.Image.merge("RGB",
                                   (r, pil.Image.new("L", cropped.size, 50), pil.Image.new("L", cropped.size, 200)))
        # loads single image into greyscale array
        data = np.array((redscale.convert('L')).resize((250, 250)).crop((25, 25, 225, 225)))

        data = data/255
        # plt.imshow(data, cmap=plt.get_cmap('gray'))
        # plt.show()
        images.append(data)

    np_images = np.array(images)
    return np_images

def load_labels(labels, n0, n):
    #load data into numpy array
    arr = np.loadtxt(labels, delimiter=",", dtype=str)
    #get just first n labels
    labels = ((arr[1:arr.shape[0],1])[n0:n])

    # seperate labels by category make list
    labels = (np.char.split(labels, sep=' ')).tolist()
    raw_labels = labels
    one_hot = MultiLabelBinarizer()
    onehot_labels = one_hot.fit_transform(labels)
    # one_hot = LabelBinarizer()
    # onehot_labels = one_hot.fit_transform(labels)
    classes = one_hot.classes_


    
    return classes, onehot_labels, raw_labels



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    n = 400
    classes, train_labels, raw_labels = load_labels("plant-pathology-2021-fgvc8/train.csv", 0, n)
    categories = len(classes)

    # data = tuple(raw_labels)
    # count = Counter(data)
    # df = pandas.DataFrame.from_dict(count, orient='index')
    # df.plot(kind='bar')
    # plt.show()

    train_images = load_resize_images("C:/Users/julpo/Desktop/train_images", 0, n)
    # # Softmax
    # model = tf.keras.Sequential(tf.keras.layers.Flatten(input_shape=(image_width, image_height)))
    # model.add(tf.keras.layers.Dense(categories, activation='softmax'))
    # model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    # 4 LAYER
    model = tf.keras.Sequential(tf.keras.layers.Flatten(input_shape=(image_width, image_height)))
    model.add(tf.keras.layers.Dense(254, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(categories, activation='relu'))
    model.compile(optimizer='adam', loss='categorical_crossentropy',  metrics=['categorical_accuracy'])

    model.fit(train_images[0:250,:,:], train_labels[0:250], epochs=20)
    print(model.summary())

    # testing_images = load_resize_images("plant-pathology-2021-fgvc8/train_images", 700, 900)
    # classes, test_labels = load_labels("plant-pathology-2021-fgvc8/train.csv", 700, 900)

    results = model.evaluate(train_images[300:400,:,:], train_labels[300:400], batch_size=64)


