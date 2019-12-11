# testing shortcut

# converting celsuis to fahrenheit
# single neuron model
# F = C * (9/5) + 32

import tensorflow as tf
import pandas as pd     # data frame manipleation
import seaborn as sns   # perform auto plot images stat plots adv matplot lib
import numpy as np      # numerical analysis
import matplotlib.pyplot as plt

temp_df = pd.read_csv('Celsius-to-Fahrenheit.csv')

print(temp_df.head(5))
print(temp_df.tail(10))
print(len(temp_df))
print(temp_df.describe())
print(temp_df.info())

# visualize data
#print(sns.scatterplot(temp_df['Celsius'], temp_df['Fahrenheit']))
#sns.scatterplot(temp_df['Celsius'], temp_df['Fahrenheit'])
#plt.scatter(temp_df['Celsius'], temp_df['Fahrenheit'], s=area, c=colors, alpha=0.5)
#colors = (0,0,0)

# display scatter plot

#colors = 'red'
#area = np.pi*3
#plt.scatter(temp_df['Celsius'], temp_df['Fahrenheit'], s=area, c=colors, alpha=0.5)
#plt.title('awdwa')
#plt.xlabel('x')
#plt.ylabel('y')
#plt.show()

x_train = temp_df['Celsius']
y_train = temp_df['Fahrenheit']

model = tf.keras.Sequential() # build our model  in a sequental faction
model.add(tf.keras.layers.Dense(units = 2, input_shape = [1]))
model.add(tf.keras.layers.Dense(units = 1, input_shape = [2]))

model.summary()

# train the model

model.compile(optimizer = tf.keras.optimizers.Adam(0.5), loss = 'mean_squared_error') 
# 0.5 = learnining rate # loss fuction

epochs_hist = model.fit(x_train, y_train, epochs = 100)

plt.plot(epochs_hist.history['loss'])
plt.show()

