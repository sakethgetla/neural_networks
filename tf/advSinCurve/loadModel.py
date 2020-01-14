#import tensorflow as tf
import pandas as pd     # data frame manipleation
#import seaborn as sns   # perform auto plot images stat plots adv matplot lib
import numpy as np      # numerical analysis
import math
import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras.layers import Dense, InputLayer
#from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

data_df = np.split(pd.read_csv('data.csv'), [2000], axis=0)

y_train = data_df[0]['y'].values
x_train = data_df[0][['x0', 'x1']].values

y_test = data_df[1]['y'].values
x_test = data_df[1][['x0', 'x1']].values

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(50)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.d1 = Dense(15, activation='tanh', input_shape = [2])
        #self.d1 = Dense(10,  activation='relu')
        self.d2 = Dense(10, activation='tanh')
        self.d3 = Dense(5, activation='tanh')
        #self.d3 = Dense(1, activation='sigmoid')
        #self.d3 = Dense(1, activation='sigmoid', use_bias=False)
        self.d5 = Dense(1)
        #self.d4 = Dense(1, activation='relu')

    def call(self, x):
        #x = self.inp(x)
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        #x = self.d4(x)
        return self.d5(x)

    #def predict(self, x):
    #    return super(MyModel, self).predict(x)

model = MyModel()
predictions = model.predict(x_test[:1])
model.load_weights('weights')


ans = [model.predict(x_test)]
# cant call predict function here? but can call in before traning

#model.save('sincurveModel.h5')
colors = 'red'
area = 7
#plt.plot(model_hist)
#plt.show()

plt.scatter(tf.transpose(x_test)[0], y_test, c=colors, alpha=0.5)
plt.scatter(tf.transpose(x_test)[0], ans, c='blue', alpha=0.5)
plt.show()


