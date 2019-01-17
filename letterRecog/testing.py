import random
import numpy as np
import math

class NeuralNet:
    def __init__(self, size, trainingData, trainingAns):
        self.trainingAns = trainingAns
        # change answers from ints to binary        
        self.trainingData = trainingData
        self.size = size
        print(self.size)
        self.NO_layers = len(size)
        print(self.NO_layers)
        self.weights = [np.random.randn( size[i], size[i-1] ) for i in range(1, self.NO_layers )]
        self.bais = [np.random.randn( size[i], 1 ) for i in range(1, self.NO_layers )]
        #self.bais = [np.random.randn( i, 1 ) for i in self.size[1:]]
        print(self.bais)
        print(self.weights)
        print(len(self.weights))
        print(len(self.weights[0]))
        print(len(self.weights[1]))
        print(sigmod(3))
        print(d_sigmod(3))
        
    def test(self, inputs):
        outputs = [np.zeros(i) for i in self.size] 
        print(outputs)
        print(inputs)
        a = []
        a =[sigmod(i) for i in inputs]
        print(a)
        print(a)
        #a = sigmod(outputs[i])
        outputs = a 
        for w, b in zip(self.weights, self.bais):
            print("inputs")
            print(outputs)
            print("weights")
            print(w)
            print("bais")
            print(b)
            print(b.transpose())
            d = [[i] for i in np.dot(w, outputs)]
            outputs = [sigmod(i) for i in np.add(d, b)]
            print(outputs)


            #print(w[0])
            #print(np.dot(w[0:3], outputs))
            #q =np.dot(w, outputs)
            #print(" ")
            #print(q)
            #print([[i] for i in q])
            #d = [[i] for i in q]
            #print(d)
            #print(" ")
            #print(np.transpose(q))
            #print(np.add(d, b))
            #print(np.dot(w, outputs)+b)
            #outputs = np.add(d, b)
#        for i in range(1, self.NO_layers):
#            for j in range(self.size[i]):
#                a = np.dot(outputs[-1], weights[i][j])
    def train(bachSize,kjhe ):
        print()

    def gradient(dataQ, dataA):
        outputs = [np.zeros(i) for i in self.size] 
        print(outputs)
        print(inputs)
        a = []
        a =[sigmod(i) for i in inputs]
        print(a)
        print(a)
        #a = sigmod(outputs[i])
        # forward
        outputs = a 
        for w, b in zip(self.weights, self.bais):
            print("inputs")
            print(outputs)
            print("weights")
            print(w)
            print("bais")
            print(b)
            d = [[i] for i in np.dot(w, outputs)]
            outputs = [sigmod(i) for i in np.add(d, b)]
            print(outputs)




def d_sigmod(x):
    a = sigmod(x)
    return a*(1-a)

def sigmod( x):
    print("sig")
    return 1/(1+ math.exp(-x))

s = [2,5,3]
net = NeuralNet(s, [0], [0])
#print(net.sigmod(3))
print(d_sigmod(3))
inputs = [2, 4] 
net.test(inputs)

 
