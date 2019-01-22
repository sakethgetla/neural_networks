
import pickle
import gzip
import numpy
import matplotlib.pyplot as plt

#f = gzip.open('../train-images-idx3-ubyte.gz', 'rb')
f = gzip.open('../data/mnist.pkl.gz', 'rb')
u = pickle._Unpickler(f)
u.encoding = 'latin1'
training_data, validation_data, test_data = u.load()
#training_data= u.load()
#print(training_data)
print(len(training_data[0]))
#50 000
print(len(validation_data[0]))
#10 000
print(len(test_data[0]))
#10 000

#d = [ [3, 2, 8], [4,7,9] ]
#afile = open(r'd.pkl', 'wb')
#pickle.dump(d, afile)
#afile.close()

def changeAns(trainingAns):
    a = np.zeros((len(trainingAns), 10))
    print(a)
    for i in range(len(trainingAns)):
        a[trainingAns[i]] = 1 
        print(i)

