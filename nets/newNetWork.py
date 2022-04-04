import matplotlib.image as mpimg
import pygame
import math
import numpy as np
import cv2

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

matxWeights = np.random.randint(9,size=(2,3))
print("matxWeights ==")
print(matxWeights)

print("matxWeights[:,0:1]) ")
print(matxWeights[:,0:1])

print("matxWeights .item(1,1)")
print(matxWeights.item(1,1))
#matxOutputs = np.random.randint(6,size=(2,3))

matxOutputs = np.zeros((2,3))
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

    print("input transpose ")
    input = np.transpose(input)
    print(input)

    identy = np.identity(2,None)
    print("identy = np.identity(2,None)")
    print(identy)
    identyWeight = np.multiply(identy,w)
    print("identyWeight = np.multiply(identy,w)")
    print(identyWeight)
    print("")
    
    print("np.multiply(identyWeight,input)")
    multiply = np.multiply(identyWeight,input)
    print(multiply)
    print("creating row")

    print(multiply)
    sa = np.vectorize(multiply)
    print("np.diag(multiply)")
    multiply = np.diag(multiply)
    print(multiply)
  
    print("addBias = np.add(multiply,b)")
    addBias = np.add(multiply,b)
    print(addBias)
    addBias[0] =  sigmoid(addBias[0])
    addBias[1] =  sigmoid(addBias[1])
    print("addBias ")
    print(addBias)
    print("tot output")
    totOut = np.sum(addBias)
    print(totOut)
#    matxOutputs[:,0] =  addBias
 #   print("matrix output" + str(matxOutputs))
    return totOut

#    matxOutputs[:,0:1] = np.reshape( addBias,(-1,1))
#    print(matxOutputs)


    #matxOutputs[:,0:1] = np.multiply(identy,input)
        

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
    return absolDist(input, matxOutputs[0,2])


def absolDist(yExpexted, yOutput):
    return ((yExpexted - yOutput)**2)/2



def d_dxSigmoid(input):
    return ((1 + math.exp(-input))**(-2))*(math.exp(-input))


def gameLoop():
    gameDisplay.fill(gray)
    currPlace = 0
    gameExit = False

    error = 0
    while not gameExit:
        pygame.display.update()
        gameDisplay.fill(gray)

        connect(neuronPos[0], neuronPos[2], matxWeights[0,0])
        connect(neuronPos[1], neuronPos[2], matxWeights[1,0])

        connect(neuronPos[0], neuronPos[3], matxWeights[0,1])
        connect(neuronPos[1], neuronPos[3], matxWeights[1,1])

        connect(neuronPos[2], neuronPos[4], matxWeights[0,2])
        connect(neuronPos[3], neuronPos[4], matxWeights[1,2])

        update(neuronPos[0],matxOutputs[0,0]) 
        update(neuronPos[1],matxOutputs[1,0])
        update(neuronPos[2],matxOutputs[0,1])
        update(neuronPos[3],matxOutputs[1,1])
        update(neuronPos[4],matxOutputs[0,2])

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
    print(yExpexted)
    matxDs[2,0] = (yExpexted - matxOutputs[0,2]) * d_dxSigmoid(matxOutputs[0,2]) * matxWeights[0,2]
    matxDs[2,1] = (yExpexted - matxOutputs[0,2]) * d_dxSigmoid(matxOutputs[0,2]) * matxWeights[1,2]

    matxDs[0,0] = d_dxSigmoid(matxDs[2,0]) * matxWeights[0,0]
    matxDs[0,1] = d_dxSigmoid(matxDs[2,0]) * matxWeights[0,1]

    matxDs[1,0] = d_dxSigmoid(matxDs[2,1]) * matxWeights[1,0]
    matxDs[1,1] = d_dxSigmoid(matxDs[2,1]) * matxWeights[1,1]
    print("matxDs")
    print(matxDs)
    #Global matxWeights 
    #Global matxWeights = matxWeights * matxDs
    print( matxWeights * np.transpose(matxDs))
    set_matxWeights()

    
def set_matxWeights():
    global matxWeights    
    matxWeights = matxWeights + ( np.transpose(matxDs) *(stepSize*(-1))  )

def calResult(train):
    print("train")
    print(train)
    matxOutputs[0,0] = sigmoid(train[0])
    matxOutputs[1,0] = sigmoid(train[1])
    print("matrix output" + str(matxOutputs))
    
    matxOutputs[0,1] = calculateOutput([matxOutputs[:,0]],matxWeights[:,0], vecBais[0])
    print("matrix output" + str(matxOutputs))
    
    matxOutputs[1,1] = calculateOutput([matxOutputs[:,0]],matxWeights[:,1],vecBais[0])
    print("matrix output" + str(matxOutputs))                                   

    matxOutputs[0,2] = calculateOutput([matxOutputs[:,1]],matxWeights[:,2],vecBais[1])
    print(matxWeights)

    print("matrix output")
    print(matxOutputs)

gameLoop()

