import tensorflow as tf
import pandas as pd     # data frame manipleation
import seaborn as sns   # perform auto plot images stat plots adv matplot lib
import numpy as np      # numerical analysis
import math
import matplotlib.pyplot as plt


def functionPredict(x):
    x +=6
    return (35*math.sin(2*x)/x)



dataSize = 3000
#theta = np.random.randint(360, size=(dataSize))
x1 = (math.pi*6 * np.random.random_sample((dataSize, )))
#sin = [math.sin(theta[i]) for i in range(dataSize)]
x0 = [t*t for t in x1]
y = [functionPredict(t) for t in x1]
#data = (theta, sin)

df = pd.DataFrame({'y' : y,
                   'x0' : x0,
                   'x1' : x1})
print(df.head(5))
print(df.describe())
print(df.info())



colors = 'red'
area = 7
plt.scatter(x1, y, c=colors, alpha=0.5)
#plt.scatter(theta, thetaSqr, c='blue', alpha=0.5)
plt.show()


df.to_csv('data.csv', index=False)

#print(data)
#print(theta)
#print(sin)
