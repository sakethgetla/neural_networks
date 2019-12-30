import tensorflow as tf
import pandas as pd     # data frame manipleation
import seaborn as sns   # perform auto plot images stat plots adv matplot lib
import numpy as np      # numerical analysis
import math

dataSize = 2000
#theta = np.random.randint(360, size=(dataSize))
theta = (math.pi*6 * np.random.random_sample((dataSize, )))
sin = [math.sin(theta[i]) for i in range(dataSize)]
data = (theta, sin)

df = pd.DataFrame({'sin' : sin,
                   'theta' : theta})
print(df.head(5))
print(df.describe())
print(df.info())



df.to_csv('data.csv', index=False)

#print(data)
#print(theta)
#print(sin)
