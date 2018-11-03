
from pynput.keyboard import Key, Controller
import matplotlib.image as mpimg
import pygame
import math
import numpy as np
import cv2
from pynput.keyboard import Key, Listener
pygame.init()
gray = (115, 115, 115)
black = (0, 0, 0)
red = (255, 0, 0)
green = (0, 155, 0)

display_width = 800
display_height = 600

width = 50
height = width

clock = pygame.time.Clock()

FPS = 30
speed = 5
dist = 150
stepSize = 0.1


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
neuronPos = []
for i in range(2):
    neuronPos.append([dist*(1) , dist*(i+1)])

for i in range(2):
    neuronPos.append([dist*(2) , dist*(i+1)])

neuronPos.append([dist*(3) , int( dist*(1.5) ) ])

training = AND
print("neruon pos ")
print(neuronPos)
print("training ")
print(training)

gameDisplay = pygame.display.set_mode((display_width, display_height))

pygame.display.set_caption('Nnet Visual')

font = pygame.font.SysFont(None, 25)


#img = mpimg.imread('aaa.png')
#print(type(img))
##print(img)

matxWeights = np.random.randint(9,size=(3,2))
print("matxWeights ==")
print(matxWeights)

print("matxWeights[1]) ")
print(matxWeights[1])

print("matxWeights .item(1,1)")
print(matxWeights.item(1,1))
#matxOutputs = np.random.randint(6,size=(2,3))

matxOutputs = np.zeros((3,2))
print("matxOutputs ==")
print(matxOutputs)

vecBais = np.array((1,1))
print(" ==")
print(vecBais)

#matxDs = np.zeros((2,3))
matxDs = np.random.randint(9,size=(3,2))
matxRnd = np.random.randint(9,size=(3,2))
print("matxDs")
print(matxDs)
#print("matxRnd")
#print(matxRnd)
#print(matxRnd*matxDs)
#print(matxRnd*5)
#print(matxDs[2,1])



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


def sigmoid(input):
    output = 1/(1+math.exp(-input))
    print("sigmoid output " + str(output))
    return(output)


def update(  pos, text):
    myfont = pygame.font.SysFont("monospace", 15)
    label = myfont.render(str(text), 1, (0, 255, 255))
    gameDisplay.blit(label, pos)
    pygame.draw.circle(gameDisplay, green, pos , 20, 3)

def connect(XY1, XY2, w ):
    pygame.draw.line(gameDisplay, red, XY2, XY1, int(round(w*2))+1 )

def calError(input):
    return absolDist(input, matxOutputs[2,0])


def absolDist(yExpexted, yOutput):
    return ((yExpexted - yOutput)**2)/2



def d_dxSigmoid(input):
    return (math.exp(-input)) / ( (1 + math.exp(-input))**(2) ) 


def gameLoop():
    gameDisplay.fill(gray)
    currPlace = 0
    gameExit = False

    error = 0
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
            print("corrpos " + str(currPlace))
            print("training " + str(training))
            if(currPlace % 4 == 3):
                currPlace += 1
                calResult(training[0][0])
                error = calError(training[0][1])
                d_ds(training[0][1])
            elif(currPlace % 4 == 2):
                currPlace += 1
                calResult(training[1][0]) 
                error = calError(training[1][1])
                d_ds(training[1][1])
            elif(currPlace % 4 == 1):
                currPlace += 1
                calResult(training[2][0])
                error = calError(training[2][1])
                d_ds(training[2][1])
            elif(currPlace % 4 == 0):
                currPlace += 1
                calResult(training[3][0])
                error = calError(training[3][1])
                print("errojjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjr")
                print(error)
                d_ds(training[3][1])
            print("error")
            print(error)
        myfont = pygame.font.SysFont("monospace", 15)
        label = myfont.render(str(error), 1, (0, 255, 255))
        gameDisplay.blit(label, [0,0] )
    pygame.quit()
    quit()

#i=0
#print("train")
#print(training[0][i])
#print("input weights")
#print(matxWeights[:,i:1])
#calculateOutput([training[0][i]],matxWeights[:,i:1],vecBais[i])
#

def d_dxCalError(yExpexted, yOutput):
    return (yExpexted - yOutput)

def d_ds(yExpexted):

    print("expexted output")
    print(yExpexted)
    temp = (yExpexted - matxOutputs[2,0]) * d_dxSigmoid(matxOutputs[2,0]) * matxWeights[2,0]
    if(temp > 50): temp = 50 
    matxDs[2,0] = temp 
    temp = (yExpexted - matxOutputs[2,0]) * d_dxSigmoid(matxOutputs[2,0]) * matxWeights[2,1]
    if(temp > 50): temp = 50 
    matxDs[2,1] = temp 

    temp = d_dxSigmoid(matxDs[2,0]) * matxWeights[0,0]
    if(temp > 50): temp = 50 
    matxDs[0,0] = temp 
    temp = d_dxSigmoid(matxDs[2,0]) * matxWeights[0,1]
    if(temp > 50): temp = 50 
    matxDs[0,1] = temp 

    temp = d_dxSigmoid(matxDs[2,1]) * matxWeights[1,0]
    if(temp > 50): temp = 50 
    matxDs[1,0] = temp 
    temp = d_dxSigmoid(matxDs[2,1]) * matxWeights[1,1]
    if(temp > 50): temp = 50 
    matxDs[1,1] = temp 

    print("matxDs")

    print(matxDs)
    #Global matxWeights 
    #Global matxWeights = matxWeights * matxDs
    
    print("matxWeights")
    print( matxWeights )
    set_matxWeights()

    
def set_matxWeights():
    global matxWeights    
    matxWeights = matxWeights * matxDs *((stepSize+1)*(1) ) 
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
    
    matxOutputs[1,1] = calculateOutput([matxOutputs[0]],matxWeights[1],vecBais[0])
    print("matrix output" + str(matxOutputs))                                   

    matxOutputs[2,0] = calculateOutput([matxOutputs[1]],matxWeights[2],vecBais[1])

    print("matrix output")
    print(matxOutputs)

gameLoop()

