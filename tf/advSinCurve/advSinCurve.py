#import tensorflow as tf
import pandas as pd     # data frame manipleation
#import seaborn as sns   # perform auto plot images stat plots adv matplot lib
import numpy as np      # numerical analysis
import math
#import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

data_df = pd.read_csv('data.csv')
print(data_df.head(5))
print(data_df.info())
print(len(data_df))

y_train = data_df['sin']
x_train = data_df[['theta', 'thetaSqr']]

#y_train = data_df['sin'].as_matrix()
#x_train = data_df['theta'].as_matrix()

#y_train = y_train.values.reshape(-1,1)
#x_train = x_train.values.reshape(-1,1)

#print(y_train)
#print(x_train)
#assert(False == True)

#y_train = data_df['sin'].as_matrix(columns=None)
#x_train = data_df['theta'].as_matrix(columns=None)

#print(x_train[0:5])
#print(y_train[0:5])
#x_train = np.asarry(x_train)

#print(x_train.as_matrix(columns=None))
#print(type(x_train))

x0= (math.pi*6 * np.random.random_sample(100))
x1= [x*x for x in x0]
x_test = [x0, x1]
y_test = [math.sin(q) for q in x0]

#x_test = x_test[..., tf.newaxis]
#x_train = x_train[..., tf.newaxis]

#assert(False == True)
#train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
#test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))

train_ds = [x_train, y_train]
train_ds = np.asarray(train_ds)
#train_ds = tf.pack(train_ds)
test_ds = [x_test, y_test]
test_ds = np.asarray(test_ds)
print(type(train_ds))
train_ds = tf.convert_to_tensor(train_ds)
print(type(train_ds))
print(shape(train_ds))
#print(test_ds)
#print(type(test_ds))
#print(train_ds)
#print("here")

assert(False == True)
#print(train_ds.info())
class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.d1 = Dense(10,  activation='relu', input_shape = [2])
        #self.d1 = Dense(10,  activation='relu')
        self.d2 = Dense(10, activation='relu')
        self.d3 = Dense(1, activation='relu')

    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        return self.d3(x)
model = MyModel()

assert(False == True)

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

    for x, y in train_ds:
        train_step(x, y)

    for test_x, test_y in test_ds:
        test_step(test_x, test_y)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch+1,
                          train_loss.result(),
                          train_accuracy.result()*100,
                          test_loss.result(),
                          test_accuracy.result()*100))


