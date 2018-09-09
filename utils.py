import tensorflow as tf
import skimage.transform
import numpy as np


def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # Wrapper for maxpolling
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='SAME')


def upsampling(x, k=2):
    # Wrapper for resizing. It actually is upsamling, which is reverse operation to pooling
    t_shape = x.get_shape()
    # print(t_shape)
    return tf.image.resize_images(x, size=[t_shape[1]*k, t_shape[2]*k])


def batch_generator(raw_image_data, batch_size):
    # this generator take a picture and random rotates it mupltiple to 90 degrees
    # angle of rotate is a lable
    angls = [0, 90, 180, 270]
    x = []
    y = []
    for i, img in enumerate(raw_image_data):
        angle = np.random.choice(angls)
        ohe_vector = np.zeros(4)
        ohe_vector[angls.index(angle)] = 1
        y.append(ohe_vector)
        transformed_img = skimage.transform.rotate(img.reshape((28, 28)), angle)
        x.append(transformed_img)
        if i % batch_size == 0:
            if i != 0:
                x = np.stack(x)
                x_out = x.reshape((-1, 28, 28, 1))
                y_out = np.stack(y)
                x = []
                y = []
                yield x_out, y_out
                
label_dict = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot',
}
