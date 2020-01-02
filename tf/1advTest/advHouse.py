#import tensorflow as tf
import pandas as pd     # data frame manipleation
#import seaborn as sns   # perform auto plot images stat plots adv matplot lib
import numpy as np      # numerical analysis
import math
#import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

temp_df = pd.read_csv('kc-house-data.csv')
print(data_df.head(5))
print(data_df.info())
print(len(data_df))

#y_train = data_df['sin']
#x_train = data_df['theta']

y_train = data_df['sin'].values()
x_train = data_df['theta'].values()
#y_train = data_df['sin'].as_matrix(columns=None)
#x_train = data_df['theta'].as_matrix(columns=None)

#print(x_train[0:5])
#print(y_train[0:5])
#x_train = np.asarry(x_train)

#print(x_train.as_matrix(columns=None))
#print(type(x_train))

x_test = (math.pi*6 * np.random.random_sample(100))
y_test = [math.sin(q) for q in x_test]

#x_test = x_test[..., tf.newaxis]
#x_train = x_train[..., tf.newaxis]

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
train_ds = tf.data.Dataset.from_tensor_slices((data_df['theta'], data_df['sin'])).batch(32)

print("here")
print(test_ds)
print(type(test_ds))
print(train_ds)
print("here")

#assert(False == True)
#print(train_ds.info())
class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.d1 = Dense(10,  activation='relu', input_shape = [1])
        #self.d1 = Dense(10,  activation='relu')
        self.d2 = Dense(10, activation='relu')
        self.d3 = Dense(1, activation='relu')

    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        return self.d3(x)
model = MyModel()

loss_obj = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='test_loss')
train_accuracy = tf.keras.metrics.MeanSquaredError(name='test_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.MeanSquaredError(name='test_accuracy')

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        predictions = model(x)
        loss = loss_obj(y, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(y, predictions)


@tf.function
def test_step(x, y):
    predictions = model(x)
    t_loss = loss_obj(y, predictions)

    test_loss(t_loss)
    test_accuracy(y, predictions)


EPOCHS = 9

for epoch in range(EPOCHS):
    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    for images, labels in train_ds:
        train_step(images, labels)

    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch+1,
                          train_loss.result(),
                          train_accuracy.result()*100,
                          test_loss.result(),
                          test_accuracy.result()*100))


