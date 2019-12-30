#import tensorflow as tf
import pandas as pd     # data frame manipleation
#import seaborn as sns   # perform auto plot images stat plots adv matplot lib
#import numpy as np      # numerical analysis
#import math
#import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

data_df = pd.read_csv('data.csv')
print(data_df.head(5))
print(data_df.info())
print(len(data_df))

y_train = data_df['sin']
x_train = data_df['theta']
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]
#train_ds = tf.data.Dataset.from_tensor_slices(
    #(x_train, y_train)).shuffle(2000).batch(32)

print(type(x_train))
#print(train_ds.info())
