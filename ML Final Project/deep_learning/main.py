import numpy as np
import tensorflow as tf
import PIL as pil
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer

batchsize = 64
image_width = 150
image_height = 150
categories = 6
classes = ['healthy', 'complex', 'scab', 'rust', 'powdery_mildew', 'frog_eye_leaf_spot']

def load_resize_images(file, n0, n):
    file_list = os.listdir(file)
    #making a subset of first 20 images
    file_list = file_list[n0:n]
    images = []
    for img in file_list:
        # loads single image into greyscale array
        data = np.array((pil.Image.open(os.path.join(file,img)).convert('L')).resize((200,200)).crop((25, 25, 175, 175)))
        # adds to list
        images.append(data/200)

    np_images = np.array(images)
    return np_images

def load_labels(labels, n0, n):
    #load data into numpy array
    arr = np.loadtxt(labels, delimiter=",", dtype=str)
    #get just first n labels
    labels = (arr[1:arr.shape[0],1])[n0:n]
    # seperate labels by category make list
    labels = (np.char.split(labels, sep = ' ')).tolist()
    one_hot = MultiLabelBinarizer()
    onehot_labels = one_hot.fit_transform(labels)
    classes = one_hot.classes_


    
    return classes, onehot_labels



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    train_images = load_resize_images("plant-pathology-2021-fgvc8/train_images", 0, 100)
    classes, train_labels = load_labels("plant-pathology-2021-fgvc8/train.csv", 0, 100)

    model = tf.keras.Sequential([tf.keras.layers.Flatten(input_shape=(image_width, image_height)),
                                 tf.keras.layers.Dense(128, activation='relu'),
                                 tf.keras.layers.Dense(128, activation='relu'),
                                 tf.keras.layers.Dense(128, activation='relu'),
                                 tf.keras.layers.Dense(128, activation='relu'),
                                 tf.keras.layers.Dense(128, activation='relu'),
                                 tf.keras.layers.Dense(categories)])

    model.compile(optimizer='adam', loss='mean_squared_error',  metrics=['mean_squared_error'])

    model.fit(train_images, train_labels, epochs=10)

    testing_images = load_resize_images("plant-pathology-2021-fgvc8/train_images", 150, 200)
    classes, test_labels = load_labels("plant-pathology-2021-fgvc8/train.csv", 150, 200)

    results = model.evaluate(testing_images, test_labels, batch_size=64)
    print("test loss, test acc:", results)

