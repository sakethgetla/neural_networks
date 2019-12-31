import tensorflow as tf
import pandas as pd     # data frame manipleation
import seaborn as sns   # perform auto plot images stat plots adv matplot lib
import numpy as np      # numerical analysis
import math
import matplotlib.pyplot as plt


data_df = pd.read_csv('data.csv')
print(data_df.head(5))
print(len(data_df))

sin = data_df['sin']
theta = data_df['theta']

colors = 'red'
area = 7
#plt.scatter(theta, sin, s=area, c=colors, alpha=0.5)
#plt.title('sin curve')
#plt.xlabel('x')
#plt.ylabel('y')
#plt.show()

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units = 10, activation = 'relu', input_shape = [1]))
#model.add(tf.keras.layers.Dense(units = 1, input_shape = [3]))
#model.add(tf.keras.layers.Dense(units = 5, activation = 'sigmoid'))
model.add(tf.keras.layers.Dense(units = 5, activation = 'relu'))
#model.add(tf.keras.layers.Dense(units = 5, activation = 'sigmoid'))
#model.add(tf.keras.layers.Dense(units = 1, activation = 'softmax'))
model.add(tf.keras.layers.Dense(units = 1))

model.summary()

#model.compile(optimizer = tf.keras.optimizers.Adam(0.05), loss = 'categorical_crossentropy')
model.compile(optimizer = tf.keras.optimizers.Adam(0.05), loss = 'mean_squared_error')

epochs_hist = model.fit(theta, sin, epochs = 100)

#plt.plot(epochs_hist.history['loss'])
#plt.show()

#Qs = math.pi/2
Qs = (math.pi*6 * np.random.random_sample(100))
print(Qs)
ans = []
corAns = []
for q in Qs:
    ans.append(model.predict([q]))
    corAns.append([math.sin(q)])

plt.scatter(Qs, ans, s=area, c=colors, alpha=0.5)
plt.scatter(Qs, corAns, s=area, c='blue', alpha=0.5)
plt.title('awdwa')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

plt.plot(epochs_hist.history['loss'])
plt.show()
#print(f"theta = {Qs}, model = {ans}, corect = {math.sin(Qs)}")
