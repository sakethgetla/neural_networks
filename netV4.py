import pygame
import math
import numpy as np

pygame.init()

gray = (115, 115, 115)
black = (0, 0, 0)
red = (255, 0, 0)
blue = (0, 0, 255)
green = (0, 255, 0)

display_width = 800
display_height = 600

width = 50
height = width

clock = pygame.time.Clock()

FPS = 10
speed = 5
dist = 150
stepSize = 3
error = 0
listError = []
listOutput = []

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

dist = 150

training = XOR 
print("training ")
print(training)

gameDisplay = pygame.display.set_mode((display_width, display_height))

pygame.display.set_caption('Nnet Visual V4')

font = pygame.font.SysFont(None, 25)

# number of neurons
nNumber = [2,3,1]

# setting the positions of neurons
neuronPos = []
for i in range(nNumber[0]):
    neuronPos.append([dist*(1) , dist*(i+1)])

for i in range(nNumber[1]):
    neuronPos.append([dist*(2) , dist*(i+1)])

neuronPos.append([dist*(3) , int( dist*(1.5) ) ])

print("neruon pos ")
print(neuronPos)

# first layer weights
W1 = np.random.rand( nNumber[0]* nNumber[1])
print("W1")
W1 -= 0.5
print(W1)


matxWeights = np.zeros((3,2))
matxWeights += np.random.randint(9,size=(3,2))

matxOutputs = np.zeros((3,2))


vecBais = np.ones((3))

matxDs = np.zeros((3,2))
matxRnd = np.random.randint(9,size=(3,2))

# second layer weights

W2 = np.random.rand(2)
W2 -= 0.5
print("W2")
print(W2)

# input layer neuron outputs
nL = np.zeros(2)

# hidden layer neuron outputs
hL = np.zeros(2)

# output of network
oL = 0.0


def calculateOutput(input,w,b):
    #w #weights 2d vector
    #b #bais
    #input #training set 2d vector

    print("b " + str(b))
    
    print("weight ")
    print(w)

    print("input ")
    print(input)
    print("input * w ")
    print((input * w) )

    print("sum ")
    print(np.sum(input * w) + b)

    print( sigmoid(np.sum(input * w) + b ))

    return sigmoid(np.sum(input * w) + b )


def calError(input):
    return absolDist(input, matxOutputs[2,0])


def absolDist(yExpexted, yOutput):
    return ((yExpexted - yOutput)**2)/2

def d_dxSigmoid(input):
    return (math.exp(-input)) / ( (1 + math.exp(-input))**(2) ) 

def connect(XY1, XY2, w ):
    if (w < 0 ) :
        pygame.draw.line(gameDisplay, blue, XY2, XY1, int(round(w*2)*(-1)))
    else: 
        pygame.draw.line(gameDisplay, red, XY2, XY1, int(round(w*2)) )

def update(  pos, text):
    myfont = pygame.font.SysFont("monospace", 15)
    label = myfont.render(str(text), 1, (0, 255, 255))
    gameDisplay.blit(label, pos)
    pygame.draw.circle(gameDisplay, green, pos , 20, 3)
     

def sigmoid(input):
    output = 1/(1+math.exp(-input))
    print("sigmoid output " + str(output))
    return(output)

def gameLoop():
    gameDisplay.fill(gray)
    currPlace = 0
    gameExit = False
    global matxWeights
    global matxDs 

    while not gameExit:
        key = "a"
        def on_press(key):
            print('{0} pressed'.format(
                key))
        pygame.display.update()
        gameDisplay.fill(gray)

        connect(neuronPos[0], neuronPos[2], matxWeights[0,0])
        connect(neuronPos[1], neuronPos[2], matxWeights[0,1])

        connect(neuronPos[0], neuronPos[3], matxWeights[1,0])
        connect(neuronPos[1], neuronPos[3], matxWeights[1,1])

        connect(neuronPos[0], neuronPos[3], matxWeights[1,0])
        connect(neuronPos[1], neuronPos[3], matxWeights[1,1])

        connect(neuronPos[2], neuronPos[4], matxWeights[2,0])
        connect(neuronPos[3], neuronPos[4], matxWeights[2,1])

        update(neuronPos[0],matxOutputs[0,0]) 
        update(neuronPos[1],matxOutputs[0,1])
        update(neuronPos[2],matxOutputs[1,0])
        update(neuronPos[3],matxOutputs[1,1])
        update(neuronPos[4],matxOutputs[2,0])

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                gameExit = True

        clock.tick(FPS)
        key = pygame.key.get_pressed()
        if key[pygame.K_d]:
            print("training " + str(training))
            global error 
            error =0 

            run(training[0])
            run(training[1])
            run(training[2])
            run(training[3])
            matxDs /= 4
            matxWeights += matxDs * stepSize *(-1)
            matxDs = np.zeros((3,2))
            error /= 4

            listError.append(error)
            print("listError")
            print(listError)
            print("listOutput")
            print(listOutput)


        myfont = pygame.font.SysFont("monospace", 15)
        label = myfont.render(str(error), 1, (0, 255, 255))
        gameDisplay.blit(label, [0,0] )
    pygame.quit()
    quit()

def d_ds(yExpexted):

    matxOnes = np.ones((3,2))
    print("matxOnes")
    print(matxOnes)
    print("expexted output")
    print(yExpexted)

    out = (matxOutputs[1,0] * matxWeights[2,0]) +  (matxOutputs[1,1] * matxWeights[2,1]) + vecBais[1]

    matxOnes[2,0] = (-1)*(yExpexted - matxOutputs[2,0]) * d_dxSigmoid(out) * matxOutputs[1,0]
    matxOnes[2,1] = (-1)*(yExpexted - matxOutputs[2,0]) * d_dxSigmoid(out) * matxOutputs[1,1]

    d_dh0 = (-1)*(yExpexted - matxOutputs[2,0]) * d_dxSigmoid(matxOutputs[2,0]) * matxWeights[2,0]
    d_dh1 = (-1)*(yExpexted - matxOutputs[2,0]) * d_dxSigmoid(matxOutputs[2,0]) * matxWeights[2,1]

    out = (matxOutputs[0,0] * matxWeights[0,0] ) + vecBais[0] + (matxOutputs[0,1] * matxWeights[0,1]) 
    matxOnes[0,0] = d_dh0 * d_dxSigmoid(out) * matxOutputs[0,0]
    matxOnes[0,1] = d_dh0 * d_dxSigmoid(out) * matxOutputs[0,1]

    out = (matxOutputs[0,0] * matxWeights[1,0] ) + vecBais[0] + (matxOutputs[0,1] * matxWeights[1,1]) 
    matxOnes[1,0] = d_dh1 * d_dxSigmoid(out) * matxOutputs[0,0]
    matxOnes[1,1] = d_dh1 * d_dxSigmoid(out) * matxOutputs[0,1]

    return matxOnes


def run(trainingSet):
    calResult(trainingSet[0])
    global error
    global d_dv

    a =  calError(trainingSet[1])
    error += a
    global matxDs
    matxDs += d_ds(trainingSet[1])
    #d_dv += d_dvs(trainingSet[1])
    d_dv += d_dvs()
    print("error")
    print(error)
    listOutput.append([matxOutputs[2,0], trainingSet])
    
def d_dvs():
    out = (matxOutputs[1,0] * matxWeights[2,0]) +  (matxOutputs[1,1] * matxWeights[2,1]) + vecBais[1]

    #d_dv[2] = (-1)*(yExpexted - matxOutputs[2,0]) * d_dxSigmoid(out)
    d_dv[1] = (-1)*(yExpexted - matxOutputs[2,0]) * d_dxSigmoid(out) 

    d_dv[0] = (-1)*(yExpexted - matxOutputs[2,0]) * d_dxSigmoid(out)


def set_matxWeights():
    global matxWeights    
    matxWeights = matxWeights + matxDs *((stepSize+1)*(-1) ) 
    print("matxWeights update")
    print( matxWeights )

def calResult(train):
    print("train")
    print(train)
    matxOutputs[0,0] = sigmoid(train[0])
    matxOutputs[0,1] = sigmoid(train[1])
    print("matrix output" + str(matxOutputs))
    
    matxOutputs[1,0] = calculateOutput([matxOutputs[0]],matxWeights[0], vecBais[0])
    print("matrix output" + str(matxOutputs))
    
    matxOutputs[1,1] = calculateOutput([matxOutputs[0]],matxWeights[1],vecBais[1])
    print("matrix output" + str(matxOutputs))                                   

    matxOutputs[2,0] = calculateOutput([matxOutputs[1]],matxWeights[2],vecBais[2])

    print("matrix output")
    print(matxOutputs)

gameLoop()
