import matplotlib.image as mpimg
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
stepSize =25 
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
neuronPos = []
for i in range(2):
    neuronPos.append([dist*(1) , dist*(i+1)])

for i in range(2):
    neuronPos.append([dist*(2) , dist*(i+1)])

neuronPos.append([dist*(3) , int( dist*(1.5) ) ])

training = XOR 
#training = AND 

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

matxWeights = np.zeros((3,2))
matxWeights += np.random.randint(10,size=(3,2))
matxWeights -= 5 

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

vecBais = np.ones((2))
d_dv = np.zeros((2))
print("vecBais")
print(vecBais)
print("d_dv")
print(d_dv)

#matxDs = np.zeros((2,3))
matxDs = np.zeros((3,2))
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
    if (w < 0 ) :
        pygame.draw.line(gameDisplay, blue, XY2, XY1, int(round(w*2)*(-1)))
    else: 
        pygame.draw.line(gameDisplay, red, XY2, XY1, int(round(w*2)) )

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

           # calResult(training[0][0])
           # error = calError(training[0][1])
           # d_ds(training[0][1])

           # calResult(training[1][0]) 
           # error = calError(training[1][1])
           # d_ds(training[1][1])

           # calResult(training[2][0])
           # error = calError(training[2][1])
           # d_ds(training[2][1])

           # calResult(training[3][0])
           # error = calError(training[3][1])
           # d_ds(training[3][1])

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

    matxOnes = np.ones((3,2))
    print("matxOnes")
    print(matxOnes)
    print("expexted output")
    print(yExpexted)

    out = (matxOutputs[1,0] * matxWeights[2,0]) +  (matxOutputs[1,1] * matxWeights[2,1])

    matxOnes[2,0] = (-1)*(yExpexted - matxOutputs[2,0]) * d_dxSigmoid(out) * matxOutputs[1,0]
    matxOnes[2,1] = (-1)*(yExpexted - matxOutputs[2,0]) * d_dxSigmoid(out) * matxOutputs[1,1]

    d_dh0 = (-1)*(yExpexted - matxOutputs[2,0]) * d_dxSigmoid(matxOutputs[2,0]) * matxWeights[2,0]
    d_dh1 = (-1)*(yExpexted - matxOutputs[2,0]) * d_dxSigmoid(matxOutputs[2,0]) * matxWeights[2,1]

    out = (matxOutputs[0,0] * matxWeights[0,0] ) + vecBais[0] + (matxOutputs[0,1] * matxWeights[0,1]) 
    matxOnes[0,0] = d_dh0 * d_dxSigmoid(out) * matxOutputs[0,0]
    matxOnes[0,1] = d_dh0 * d_dxSigmoid(out) * matxOutputs[0,1]

    out = (matxOutputs[0,0] * matxWeights[1,0] ) + vecBais[1] + (matxOutputs[0,1] * matxWeights[1,1]) 
    matxOnes[1,0] = d_dh1 * d_dxSigmoid(out) * matxOutputs[0,0]
    matxOnes[1,1] = d_dh1 * d_dxSigmoid(out) * matxOutputs[0,1]

    return matxOnes

    #temp = (yExpexted - matxOutputs[2,0]) * d_dxSigmoid(matxOutputs[2,0]) * matxWeights[2,0]
    #if(temp > 50): temp = 50 
    #matxOnes[2,0] = temp 
    #temp = (yExpexted - matxOutputs[2,0]) * d_dxSigmoid(matxOutputs[2,0]) * matxWeights[2,1]
    #if(temp > 50): temp = 50 
    #matxOnes[2,1] = temp 

    #temp = d_dxSigmoid(matxOnes[2,0]) * matxWeights[0,0]
    #if(temp > 50): temp = 50 
    #matxOnes[0,0] = temp 
    #temp = d_dxSigmoid(matxOnes[2,0]) * matxWeights[0,1]
    #if(temp > 50): temp = 50 
    #matxOnes[0,1] = temp 

    #temp = d_dxSigmoid(matxOnes[2,1]) * matxWeights[1,0]
    #if(temp > 50): temp = 50 
    #matxOnes[1,0] = temp 
    #temp = d_dxSigmoid(matxOnes[2,1]) * matxWeights[1,1]
    #if(temp > 50): temp = 50 
    #matxOnes[1,1] = temp 

    
#    matxOnes[2,0] =  (yExpexted - matxOutputs[2,0]) * d_dxSigmoid(matxOutputs[2,0]) * matxWeights[2,0]
#    
#    matxOnes[2,1] =  (yExpexted - matxOutputs[2,0]) * d_dxSigmoid(matxOutputs[2,0]) * matxWeights[2,1]
#
#    
#    matxOnes[0,0] =  d_dxSigmoid(matxOnes[2,0]) * matxWeights[0,0]
#    
#    matxOnes[0,1] =  d_dxSigmoid(matxOnes[2,0]) * matxWeights[0,1]
#
#    
#    matxOnes[1,0] =  d_dxSigmoid(matxOnes[2,1]) * matxWeights[1,0]

def run(trainingSet):
    global error
    global d_dv
    global matxDs

    calResult(trainingSet[0])
    a =  calError(trainingSet[1])
    error += a
    matxDs += d_ds(trainingSet[1])
    d_dv += d_dvs(trainingSet[1])
    #d_dv += d_dvs()
    print("error")
    print(error)
    listOutput.append([matxOutputs[2,0], trainingSet])
    
def d_dvs(yExpexted):
    out = (matxOutputs[1,0] * matxWeights[2,0]) +  (matxOutputs[1,1] * matxWeights[2,1]) + vecBais[1]
    vex = np.zeros(2)
    vex[0] = (-1)*(yExpexted - matxOutputs[2,0]) * d_dxSigmoid(out)
    vex[1] = (-1)*(yExpexted - matxOutputs[2,0]) * d_dxSigmoid(out) 
    return vex

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

    matxOutputs[2,0] = calculateOutput([matxOutputs[1]],matxWeights[2],0)

    print("matrix output")
    print(matxOutputs)

gameLoop()

