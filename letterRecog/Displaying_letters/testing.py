import pickle
import gzip
import numpy
import matplotlib.pyplot as plt
from PIL import Image

# show a number picked from the traning data
#i = the training_data number

f = gzip.open('../data/mnist.pkl.gz', 'rb')
u = pickle._Unpickler(f)
u.encoding = 'latin1'
training_data, validation_data, test_data = u.load()

#print(training_data)
#print(training_data[0][0])
print(len(training_data[0][0]))
print(len(training_data[0]))
print(len(training_data[1]))
print(len(training_data))


i=4

img = training_data[0][i].copy()
ans = training_data[1][i]
img *= 225

#print(img)
print(len(img))
print(ans)

img.resize((28,28))
plt.imshow(img )
plt.show()
