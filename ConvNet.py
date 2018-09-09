from tensorflow.examples.tutorials.mnist import input_data
from atfnlg.tmp.utils_ import *


# https://www.datacamp.com/community/tutorials/cnn-tensorflow-python
#TODO:
# Put saver
# Interface that could perfrom image representation vectors


def conv_net(x, weights, biases):

    # here we call the conv2d function we had defined above and pass the input image x, weights wc1 and bias bc1.
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 14*14 matrix.
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    # here we call the conv2d function we had defined above and pass the input image x, weights wc2 and bias bc2.
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 7*7 matrix.
    conv2 = maxpool2d(conv2, k=2)

    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])

    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 4*4.
    conv3 = maxpool2d(conv3, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Output, class prediction
    # finally we multiply the fully connected layer with the weights and add a bias term.
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out


def main():
    data = input_data.read_data_sets('data/fashion', one_hot=True)
    print("Training set (images) shape: {shape}".format(shape=data.train.images.shape))
    print("Training set (labels) shape: {shape}".format(shape=data.train.labels.shape))
    # Shapes of test set
    print("Test set (images) shape: {shape}".format(shape=data.test.images.shape))
    print("Test set (labels) shape: {shape}".format(shape=data.test.labels.shape))

    train_X = data.train.images.reshape(-1, 28, 28, 1)
    test_X = data.test.images.reshape(-1, 28, 28, 1)

    train_y = data.train.labels
    test_y = data.test.labels

    training_iters = 200
    learning_rate = 0.001
    batch_size = 128

    # MNIST data input (img shape: 28*28)
    n_input = 28

    # MNIST total classes (0-9 digits)
    n_classes = 4

    x = tf.placeholder("float", [None, 28, 28, 1])
    y = tf.placeholder("float", [None, n_classes])

    weights = {
        'wc1': tf.get_variable('W0', shape=(3, 3, 1, 32), initializer=tf.contrib.layers.xavier_initializer()),
        'wc2': tf.get_variable('W1', shape=(3, 3, 32, 64), initializer=tf.contrib.layers.xavier_initializer()),
        'wc3': tf.get_variable('W2', shape=(3, 3, 64, 128), initializer=tf.contrib.layers.xavier_initializer()),
        'wd1': tf.get_variable('W3', shape=(4 * 4 * 128, 128), initializer=tf.contrib.layers.xavier_initializer()),
        'out': tf.get_variable('W6', shape=(128, n_classes), initializer=tf.contrib.layers.xavier_initializer()),
    }
    biases = {
        'bc1': tf.get_variable('B0', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
        'bc2': tf.get_variable('B1', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
        'bc3': tf.get_variable('B2', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
        'bd1': tf.get_variable('B3', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
        'out': tf.get_variable('B4', shape=(n_classes), initializer=tf.contrib.layers.xavier_initializer()),
    }

    pred = conv_net(x, weights, biases)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        train_loss = []
        test_loss = []
        train_accuracy = []
        test_accuracy = []
        tf.summary.scalar("Model loss", cost)
        tf.summary.scalar("Model accuracy", accuracy)
        merged_summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter('./Output', sess.graph)
        step = 0
        for i in range(training_iters):
            for batch_x, batch_y in batch_generator(data.train.images, 128):
                # Run optimization op (backprop).
                # Calculate batch loss and accuracy
                loss, acc, opt = sess.run([cost, accuracy, optimizer], feed_dict={x: batch_x,
                                                                                  y: batch_y})
                tb_log = sess.run(merged_summary_op, feed_dict={x: batch_x,
                                                                y: batch_y})
                summary_writer.add_summary(tb_log, step)
                step += 1
            # Calculate accuracy for all 10000 mnist test images
            test_acc, valid_loss = sess.run([accuracy, cost], feed_dict={x: test_X, y: test_y})
            # train_loss.append(loss)
            # test_loss.append(valid_loss)
            # train_accuracy.append(acc)
            # test_accuracy.append(test_acc)
            # print("Testing Accuracy:", "{:.5f}".format(test_acc))
        summary_writer.close()

        ''' original
    with tf.Session() as sess:
        sess.run(init)
        train_loss = []
        test_loss = []
        train_accuracy = []
        test_accuracy = []
        tf.summary.scalar("Model loss", cost)
        tf.summary.scalar("Model accuracy", accuracy)
        merged_summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter('./Output', sess.graph)
        step = 0
        for i in range(training_iters):
            for batch in range(len(train_X) // batch_size):
                batch_x = train_X[batch * batch_size:min((batch + 1) * batch_size, len(train_X))]
                batch_y = train_y[batch * batch_size:min((batch + 1) * batch_size, len(train_y))]
                # Run optimization op (backprop).
                # Calculate batch loss and accuracy
                loss, acc, opt = sess.run([cost, accuracy, optimizer], feed_dict={x: batch_x,
                                                     y: batch_y})
                tb_log = sess.run(merged_summary_op, feed_dict={x: batch_x,
                                                                  y: batch_y})
                summary_writer.add_summary(tb_log, step)
                step += 1
            # Calculate accuracy for all 10000 mnist test images
            test_acc, valid_loss = sess.run([accuracy, cost], feed_dict={x: test_X, y: test_y})
            #train_loss.append(loss)
            #test_loss.append(valid_loss)
            #train_accuracy.append(acc)
            #test_accuracy.append(test_acc)
            #print("Testing Accuracy:", "{:.5f}".format(test_acc))
        summary_writer.close()
        '''

if __name__ == "__main__":
    main()
