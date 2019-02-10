from newNetWork import NeuralNet
import pdb
import random
import numpy as np
import math


XOR = [
    [[0, 0], [0]],
    [[0, 1], [1]],
    [[1, 0], [1]],
    [[1, 1], [0]]
]
AND = [
    [[0, 0], [0]],
    [[0, 1], [0]],
    [[1, 0], [0]],
    [[1, 1], [1]]
]


s = [2,2,1]
testSize = 9
bachSize = 1
stepSize = 8
delta_w  = [np.zeros((a,b)) for a,b in zip(s[1:], s[:-1])]
delta_b  = [np.zeros((a,1)) for a,b in zip(s[1:], s[:-1])]
print(delta_w)
print(delta_b)
delta_b = np.asarray(delta_b)
print(delta_b)
pdb.set_trace()
#delta_b = []

pet = NeuralNet(s, [0], [0,1])

wow = stepSize/bachSize
for i in range(testSize):
    for j in range(bachSize):
        print(j)
        for inputs, output in AND:
            print("inhjhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhputs")
            print("inputs")
            print(inputs)
            print("output")
            print(output)
            jacoben_w, jacoben_b =  pet.gradient(inputs, output)
            delta_w = np.add(delta_w,jacoben_w)
            print("delta_b")
            print(delta_b)
            print("jacoben_b")
            print(type(jacoben_b))
#print(len(delta_b))
#print("jacoben_b")
#print(jacoben_b)
#print(len(jacoben_b))
            jacoben_b = np.asarray(jacoben_b)
            print("jacoben_b")
            print(type(jacoben_b))
            print(jacoben_b)
            delta_b = np.add(delta_b,jacoben_b)
            print("delta_w")
            print(delta_w)
            print("delta_b")
            print(delta_b)

    print("delta_w")
    print(delta_w)
    print("delta_b")
    print(delta_b)
    pet.weights = np.add(pet.weights, [wow * w for w in delta_w])
    #pet.bais = np.asarray(pet.bais)
    pet.bais =  np.add(pet.bais, [wow *b for b in delta_b])
    totError = 0
    for inp, out in AND:
        totError += pet.error(out, pet.test(inp))
    print(totError)
