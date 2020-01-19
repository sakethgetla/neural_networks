import pandas as pd     # data frame manipleation
#import seaborn as sns   # perform auto plot images stat plots adv matplot lib
import numpy as np      # numerical analysis
import math
import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras.layers import Dense, InputLayer, Flatten, Conv2D
#from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
from pdb import set_trace as bp

#data_df = pd.read_csv('mnist.csv')
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

#bp()
# Add a channels dimension
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]
bp()

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(100000).batch(50)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(50)

class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        #self.input = InputLayer(input_tensor=np.shape(x_train[0]))
        #self.inp= InputLayer()
        #self.inp= InputLayer(input_tensor=x_train[0])
        #self.d1 = Dense(5, activation='sigmoid')
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='tanh')
        #self.d1 = Dense(15, activation='tanh', input_shape = [28, 28, 1])
        #self.d1 = Dense(10,  activation='relu')
        self.d2 = Dense(10, activation='softmax')
        #self.d3 = Dense(1, activation='tanh')
        #self.d3 = Dense(1, activation='sigmoid')
        #self.d3 = Dense(1, activation='sigmoid', use_bias=False)
        #self.d3 = Dense(1)
        #self.d4 = Dense(1, activation='relu')

    def call(self, x):
        #x = self.inp(x)
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        #x = self.d2(x)
        #x = self.d3(x)
        #x = self.d4(x)
        return self.d2(x)

    #def predict(self, x):
    #    return super(MyModel, self).predict(x)


#assert(False == True)

model = MyModel()
loss_obj = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='test_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')



@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        #print("x = {x} y = {y}")
        #tf.print(x, output_stream=sys.stdout)
        #print(f"x = {x} y = {y}")
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




EPOCHS = 5
model_hist = []

for epoch in range(EPOCHS):
    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    #for x, y in zip(x_train, y_train):
    #bp()
    model.predict(x_train[:1])
    for x, y in train_ds:
        train_step(x, y)

    #for x, y in zip(x_test, y_test):
    for x, y in test_ds:
        test_step(x, y)
    model_hist.append(train_loss.result())

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch+1,
                          train_loss.result(),
                          train_accuracy.result()*100,
                          test_loss.result(),
                          test_accuracy.result()*100))


model.save_weights('weights', save_format='tf')

ans = [model.predict(x_test)]
# cant call predict function here? but can call in before traning

#model.save('sincurveModel.h5')
colors = 'red'
area = 7
plt.plot(model_hist)
plt.show()




