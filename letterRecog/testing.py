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
        #self.bais = [np.random.randn( i, 1 ) for i in self.size[1:]]
        print(self.bais)
        print(self.weights)
        print(len(self.weights))
        print(len(self.weights[0]))
        print(len(self.weights[1]))
        print(sigmod(3))
        print(d_sigmod(3))
        
#    def test(self, inputs):
#        outputs = [np.zeros(i) for i in self.size] 
#        print(outputs)
#        print(inputs)
#        a = []
#        a =[sigmod(i) for i in inputs]
#        print(a)
#        print(a)
#        #a = sigmod(outputs[i])
#        outputs = a 
#        for w, b in zip(self.weights, self.bais):
#            print("inputs")
#            print(outputs)
#            print("weights")
#            print(w)
#            print("bais")
#            print(b)
#            print(b.transpose())
#            d = [[i] for i in np.dot(w, outputs)]
#            outputs = [sigmod(i) for i in np.add(d, b)]
#            print(outputs)
#

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

    def test(self, inputs):
        outputs = [np.zeros(i) for i in self.size] 
        print(outputs)
        print(inputs)
        #a = []
        #a =[sigmod(i) for i in inputs]
        #print(a)
        #print(a)
        zs = []
        z = [ i for i in inputs]  
        zs.append(z)
        #a = sigmod(outputs[i])
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
            
            #z = [[i] for i in np.dot(w, outputs)]
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
        #a = []
        #a =[sigmod(i) for i in inputs]
        #print(a)
        #print(a)
        zs = []
        z = [ i for i in dataQ]  
        zs.append(z)
        #a = sigmod(outputs[i])
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
            
            #z = [[i] for i in np.dot(w, outputs)]
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
        jacoben_w = [np.zeros((self.size[i], self.size[i-1])) for i in range(1, self.NO_layers) ]
        jacoben_b = [np.zeros((1, self.size[i])) for i in range(1, self.NO_layers) ]
        print("jacoben_w")
        print(jacoben_w)
        print("jacoben_b")
        print(jacoben_b)

        temp = [ (y -o)* d_sigmod(z) for y,o,z in zip(dataA, activations[-1], zs[-1])]
        print("temp")
        print(temp)
        #np.multiply

        #jacoben_w[-1] = [activations[-2] * t for t in temp ]
        jacoben_w[-1] = [ np.multiply(t, activations[-2]) for t in temp]
        jacoben_b[-1] = temp
        print("jacoben_w")
        print(jacoben_w[-1])
        print("jacoben_b")
        print(jacoben_b[-1])
        for i in range(1, self.NO_layers -1):
            print(i)
            temp = [temp* w* d_sigmod(z) for w, z in zip(w[-i].transpose(), zs[-i-1])]
            print("temp")
            print(temp)
            jacoben_w[-i-1] = [ temp* out for out in activations[-i -2]]
            jacoben_b[-i-1] = [temp]
            print("jacoben_w")
            print(jacoben_w[-i-1])
            print("jacoben_b")
            print(jacoben_b[-i-1])

        print("jacoben_w")
        print(jacoben_w)
        print("jacoben_b")
        print(jacoben_b)



#        outputs = [np.zeros(i) for i in self.size] 
#        print(outputs)
#        print(inputs)
#        a = []
#        a =[sigmod(i) for i in inputs]
#        print(a)
#        print(a)
#        #a = sigmod(outputs[i])
#        # forward
#        outputs = a 
#        for w, b in zip(self.weights, self.bais):
#            print("inputs")
#            print(outputs)
#            print("weights")
#            print(w)
#            print("bais")
#            print(b)
#            d = [[i] for i in np.dot(w, outputs)]
#            outputs = [sigmod(i) for i in np.add(d, b)]
#            print(outputs)




def d_sigmod(x):
    a = sigmod(x)
    return a*(1-a)

def sigmod( x):
    print("sig")
    return 1/(1+ math.exp(-x))

s = [2,5,3]
net = NeuralNet(s, [0], [0,1])
#print(net.sigmod(3))
print(d_sigmod(3))
inputs = [2, 4] 
net.test(inputs)
net.gradient(inputs,[0,0,0])

 
