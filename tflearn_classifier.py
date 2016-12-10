import tflearn
import tensorflow as tf
import numpy as np
from tflearn.data_utils import to_categorical

# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39], [177, 70, 40], [159, 55, 37], [171, 75, 42],
     [181, 85, 43], [170, 52, 39]]

# 0 - for female, 1 - for male
Y = [1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0]

# encode to one-hot vector and shuffle
data = np.column_stack((X, to_categorical(Y, nb_classes=2)))
np.random.shuffle(data)

# Split into train and test set
X_train, Y_train = data[:8, :3], data[:8, 3:]
X_test, Y_test = data[8:, :3], data[8:, 3:]

# Build neural network
net = tflearn.input_data(shape=[None, 3])
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net)

# fix for tflearn with TensorFlow 12:
col = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
for x in col:
    tf.add_to_collection(tf.GraphKeys.VARIABLES, x)

# Define model
model = tflearn.DNN(net)
# Start training (apply gradient descent algorithm)
model.fit(X_train, Y_train, n_epoch=100, show_metric=True)

score = model.evaluate(X_test, Y_test)
print('Training test score', score)

test_male = [176, 78, 42]
test_female = [170, 52, 38]
print('Test male: ', model.predict([test_male])[0])
print('Test female:', model.predict([test_female])[0])
