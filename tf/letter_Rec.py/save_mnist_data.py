import pandas as pd     # data frame manipleation
#import seaborn as sns   # perform auto plot images stat plots adv matplot lib
import numpy as np      # numerical analysis
import math
import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras.layers import Dense, InputLayer
#from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

mnist = tf.keras.datasets.mnist

train, test = mnist.load_data()
#x_train, x_test = x_train / 255.0, x_test / 255.0

df = pd.DataFrame({'train' : train,
                   'test' : test})

#df = pd.DataFrame({'x_train' : x_train,
#                   'y_train' : y_train,
#                   'x_test' : x_test,
#                   'y_test' : y_test})

