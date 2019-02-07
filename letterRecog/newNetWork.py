import random
import numpy as np
import math

class NeuralNet:
    def __init__(self, size, trainingData, trainingAns):
        self.trainingAns = trainingAns
        a = np.zeros((len(trainingAns), 10))
        print(a)
        for i in range(len(trainingAns)):
            print(i)

        # change answers from ints to binary        
        self.trainingData = trainingData
        self.size = size
        print(self.size)
        self.NO_layers = len(size)
        print(self.NO_layers)
        self.weights = [np.random.randn( size[i], size[i-1] ) for i in range(1, self.NO_layers )]
        self.bais = [np.random.randn( size[i], 1 ) for i in range(1, self.NO_layers )]
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
        zs = []
        z = [ i for i in inputs]  
        zs.append(z)
        outputs = [sigmod(i) for i in inputs ]
        print("outputs")
        print(outputs)
        activations = []
        activations.append(outputs)
        for w, b in zip(self.weights, self.bais):
            print("inputs")
            print(outputs)
            print("weights")
            print(w)
            print("bais")
            print(b)
            print(b.transpose())
            z = [[i] for i in np.dot(w, outputs)]
            print("z")
            print(z)
            z = np.add(z, b)
            print()
            print("z")
            print(z)
            print()
            zs.append(z)
            outputs = [sigmod(i) for i in z]
            print(outputs)
            activations.append(outputs)

        print(zs)
        print(activations)

    def gradient(self, dataQ, dataA):
        outputs = [np.zeros(i) for i in self.size] 
        print(outputs)
        print(dataQ)
        zs = []
        z = [ i for i in dataQ]  
        zs.append(z)
        outputs = [sigmod(i) for i in inputs ]
        print("outputs")
        print(outputs)
        activations = []
        activations.append(outputs)
        # forward
        for w, b in zip(self.weights, self.bais):
            print("inputs")
            print(outputs)
            print("weights")
            print(w)
            print("bais")
            print(b)
            print(b.transpose())
            
            z = [[i] for i in np.dot(w, outputs)]
            print("z")
            print(z)
            z = np.add(z, b)
            print()
            print("z")
            print(z)
            print()
            zs.append(z)
            outputs = [sigmod(i) for i in z]
            print(outputs)
            activations.append(outputs)

        print(zs)
        print(activations)
        #back prop
        jacoben_w = [np.zeros((self.size[i], self.size[i-1])) for i in range(1, self.NO_layers) ]
        jacoben_b = [np.zeros((1, self.size[i])) for i in range(1, self.NO_layers) ]
        print("jacoben_w")
        print(jacoben_w)
        print(np.shape(jacoben_w))
        print("jacoben_b")
        print(jacoben_b)

        temp = [d_error(y,o) * d_sigmod(z) for y,o,z in zip(dataA, activations[-1], zs[-1])]
        #temp = [(d_error(y,o) * d_sigmod(z) for y,o,z in zip(dataA, activations[-1], zs[-1])]
        print("temp")
        print(temp)
        jacoben_w[-1] = [ np.multiply(t, activations[-2]) for t in temp]
        jacoben_b[-1] = temp
        print("jacoben_w")
        print(jacoben_w[-1])
        print("jacoben_b")
        print(jacoben_b[-1])
        for i in range(1, self.NO_layers -1):
            print(i)

            print("temp")
            print(temp)
            print(np.shape(temp))
            print("self.weights[-i].transpose()")
            print(self.weights[-i].transpose())
            print(" zs[-i-1]")
            print( zs[-i-1])

            temp = [np.dot(temp, w )*d_sigmod(z) for w,z in zip(self.weights[-i].transpose(), zs[-i-1])]
            print("temp1")
            print(temp)
            print(np.shape(temp))

            jacoben_w[-i-1] = [np.multiply(t, activations[-i-2]) for t in temp]
            jacoben_b[-i-1] = [temp]
            print("jacoben_w[-i-1]")
            print(jacoben_w[-i-1])
            print("jacoben_b[-i-1]")
            print(jacoben_b[-i-1])

        print()
        print()
        print()
        print("jacoben_w")
        print(jacoben_w)
        print(np.shape(jacoben_w))
        print("jacoben_b")
        print(jacoben_b)
        print(np.shape(jacoben_b))
        print("done!!!")



    def train(bachSize,trainingData_Q, trainingData_A ):
        print()
        dataQ = tr
        for dataQ, dataA in zip(trainingData_Q, trainingData_A):
            jacoben_w, jacoben_b = gradient(dataQ, dataA)
            sum_Ws += jacoben_w
            sum_bs += jacoben_b



def d_error(y,o):
    print("error")
    return y-o

def d_sigmod(x):
    a = sigmod(x)
    return a*(1-a)

def sigmod(x):
    print("sig")
    return 1/(1+ math.exp(-x))

#s = [3,5,3]
s = [2,2,1]
net = NeuralNet(s, [0], [0,1])
print(d_sigmod(3))
inputs = [ 4,3] 
net.test(inputs)
net.gradient(inputs,[1])

 
