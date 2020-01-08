import pandas as pd     # data frame manipleation
#import seaborn as sns   # perform auto plot images stat plots adv matplot lib
import numpy as np      # numerical analysis
import math
import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras.layers import Dense, InputLayer
#from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model


class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.d1 = Dense(15, activation='tanh', input_shape = [2])
        self.d2 = Dense(10, activation='tanh')
        self.d3 = Dense(5, activation='tanh')
        self.d5 = Dense(1)

    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        return self.d5(x)

    def predict(self, x):
        return super(MyModel, self).predict(x)

def initalise():
    data_df = np.split(pd.read_csv('data.csv'), [2000], axis=0)

    y_train = data_df[0]['y'].values
    x_train = data_df[0][['x0', 'x1']].values

    y_test = data_df[1]['y'].values
    x_test = data_df[1][['x0', 'x1']].values

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(50)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
    model = MyModel()

    loss_obj = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(0.0001)

    #train_loss = tf.keras.metrics.Mean(name='test_loss')
    #train_accuracy = tf.keras.metrics.MeanSquaredError(name='test_accuracy')

    #test_loss = tf.keras.metrics.Mean(name='test_loss')
    #test_accuracy = tf.keras.metrics.MeanSquaredError(name='test_accuracy')
    model.predict(x_train[:2])
    return train_ds, test_ds, model, loss_obj, optimizer

@tf.function
def train_step(x, y, model, loss_obj, optimizer):
    with tf.GradientTape() as tape:
        #print("x = {x} y = {y}")
        #tf.print(x, output_stream=sys.stdout)
        #print(f"x = {x} y = {y}")
        predictions = model(x)
        loss = loss_obj(y, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    #train_loss(loss)

@tf.function
def test_step(x, y, model):
    predictions = model(x)
    #t_loss = loss_obj(y, predictions)

    #test_loss(t_loss)
    #test_accuracy(y, predictions)
    return predictions

#EPOCHS = 5
#model_hist = []

def runEpoch(train_ds, test_ds, model, loss_obj, optimizer):
    # Reset the metrics at the start of the next epoch
    #train_loss.reset_states()
    #train_accuracy.reset_states()
    #test_loss.reset_states()
    #test_accuracy.reset_states()

    #for x, y in zip(x_train, y_train):
    for x, y in train_ds:
        train_step(x, y, model, loss_obj, optimizer)

    #for x, y in zip(x_test, y_test):
    for x, y in test_ds:
        predictions = test_step(x, y, model)
    #model_hist.append(train_loss.result())

    #template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    #print(template.format(epoch+1,
    #                      train_loss.result(),
    #                      train_accuracy.result()*100,
    #                      test_loss.result(),
    #                      test_accuracy.result()*100))
    return predictions, model, loss_obj, optimizer

#for epoch in range(EPOCHS):
#    # Reset the metrics at the start of the next epoch
#    train_loss.reset_states()
#    train_accuracy.reset_states()
#    test_loss.reset_states()
#    test_accuracy.reset_states()
#
#    #for x, y in zip(x_train, y_train):
#    model.predict(x_train[:2])
#    for x, y in train_ds:
#        train_step(x, y)
#
#    #for x, y in zip(x_test, y_test):
#    for x, y in test_ds:
#        test_step(x, y)
#    model_hist.append(train_loss.result())
#
#    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
#    print(template.format(epoch+1,
#                          train_loss.result(),
#                          train_accuracy.result()*100,
#                          test_loss.result(),
#                          test_accuracy.result()*100))


#ans = [model.predict(x_test)]
#colors = 'red'
#area = 7
#plt.plot(model_hist)
#plt.show()
#
#plt.scatter(tf.transpose(x_test)[0], y_test, c=colors, alpha=0.5)
#plt.scatter(tf.transpose(x_test)[0], ans, c='blue', alpha=0.5)
#plt.show()
