# visual neural network that learns from logic gates 2 inputs and 1 output
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

width = 10
height = width

clock = pygame.time.Clock()

FPS = 1000
#speed = 5
dist = 150
stepSize =5 
error = 4
listError = []
listOutput = []
#runningTime = 5000

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

SPC = [
    [[0, 0], [0]],
    [[0, 1], [1]],
    [[1, 0], [0]],
    [[1, 1], [1]]
]


neuronPos = []
for i in range(2):
    neuronPos.append([dist*(1) , dist*(i+1)])

for i in range(2):
    neuronPos.append([dist*(2) , dist*(i+1)])

neuronPos.append([dist*(3) , int( dist*(1.5) ) ])

#training = SPC 
#training = AND 
training = XOR 

print("neruon pos ")
print(neuronPos)
print("training ")
print(training)

gameDisplay = pygame.display.set_mode((display_width, display_height))

pygame.display.set_caption('Nnet Visual')

font = pygame.font.SysFont(None, 25)



matxWeights = np.zeros((3,2))
matxWeights += np.random.randint(10,size=(3,2))
matxWeights -= 5 

print("matxWeights ==")
print(matxWeights)

print("matxWeights[1]) ")
print(matxWeights[1])

print("matxWeights .item(1,1)")
print(matxWeights.item(1,1))

matxOutputs = np.zeros((3,2))
print("matxOutputs ==")
print(matxOutputs)

vecBais = [-1,1]
d_dv = np.zeros((2))
print("vecBais")
print(vecBais)
print("d_dv")
print(d_dv)

matxDs = np.zeros((3,2))
matxRnd = np.random.randint(9,size=(3,2))
print("matxDs")
print(matxDs)



def calculateOutput(input,w,b):

    print("b " + str(b))
    
    print("weight ")
    print(w)

    print("input ")
    print(input)
    print("input * w ")
    print((input * w) )

    print("sum ")
    print(np.sum(input * w) + b)
    y =  sigmoid(np.sum(input * w) + b )
    print(y)

    return y


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
    global vecBais 
    global d_dv 
    global error 
    counter = 0 

    while not gameExit:
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
        if key[pygame.K_d] or error > 0.002:
            counter +=1 
            print("training " + str(training))
            error =0 

            run(training[0])
            run(training[1])
            run(training[2])
            run(training[3])
            matxDs /= 4
            matxWeights += matxDs * stepSize *(-1)
            matxDs = np.zeros((3,2))
            d_dv /=4 
            vecBais += d_dv * stepSize *(-1)
            d_dv = np.zeros((2))
            error /= 4

            listError.append(error)
            print("listError")
            print("listOutput")
            print(listOutput[-4:])


        myfont = pygame.font.SysFont("monospace", 15)
        label = myfont.render(str(error), 1, (0, 255, 255))
        gameDisplay.blit(label, [0,0] )
    print("matxWeights ==")
    print(matxWeights)
    print("vecBais")
    print(vecBais)
    print("error")
    print(error)
    pygame.quit()
    quit()


def d_dxCalError(yExpexted, yOutput):
    # cross entropy cost function
    return (yExpexted/yOutput)+((yExpexted-1)/(1-yOutput))
    #return (yExpexted - yOutput)

def d_ds(yExpexted):

    matxOnes = np.ones((3,2))
    print("matxOnes")
    print(matxOnes)
    print(type(matxOnes))
    print("expexted output")
    print(yExpexted)
    print(type(yExpexted[0]))
    print("matxOutputs[2,0]")
    print(matxOutputs[2,0])
    print(type(matxOutputs[2,0]))

    out = (matxOutputs[1,0] * matxWeights[2,0]) +  (matxOutputs[1,1] * matxWeights[2,1])

    matxOnes[2,0] = (-1)*d_dxCalError(yExpexted[0] , matxOutputs[2,0]) * d_dxSigmoid(out) * matxOutputs[1,0]
    matxOnes[2,1] = (-1)*d_dxCalError(yExpexted[0] , matxOutputs[2,0]) * d_dxSigmoid(out) * matxOutputs[1,1]

    d_dh0 = (-1)*d_dxCalError(yExpexted[0] , matxOutputs[2,0]) * d_dxSigmoid(out) * matxWeights[2,0]
    d_dh1 = (-1)*d_dxCalError(yExpexted[0] , matxOutputs[2,0]) * d_dxSigmoid(out) * matxWeights[2,1]

    out = (matxOutputs[0,0] * matxWeights[0,0] ) + vecBais[0] + (matxOutputs[0,1] * matxWeights[0,1]) 
    matxOnes[0,0] = d_dh0 * d_dxSigmoid(out) * matxOutputs[0,0]
    matxOnes[0,1] = d_dh0 * d_dxSigmoid(out) * matxOutputs[0,1]

    out = (matxOutputs[0,0] * matxWeights[1,0] ) + vecBais[1] + (matxOutputs[0,1] * matxWeights[1,1]) 
    matxOnes[1,0] = d_dh1 * d_dxSigmoid(out) * matxOutputs[0,0]
    matxOnes[1,1] = d_dh1 * d_dxSigmoid(out) * matxOutputs[0,1]

    return matxOnes


def run(trainingSet):
    global error
    global d_dv
    global matxDs
    
    calResult(trainingSet[0])
    a =  calError(trainingSet[1])
    error += a
    matxDs += d_ds(trainingSet[1])
    d_dv += d_dvs(trainingSet[1])
    print("error")
    print(error)
    listOutput.append([matxOutputs[2,0], trainingSet])
    
def d_dvs(yExpexted):

    out = (matxOutputs[1,0] * matxWeights[2,0]) +  (matxOutputs[1,1] * matxWeights[2,1])
    d_dh0 = (-1)*d_dxCalError(yExpexted[0] , matxOutputs[2,0]) * d_dxSigmoid(out) * matxWeights[2,0]
    d_dh1 = (-1)*d_dxCalError(yExpexted[0] , matxOutputs[2,0]) * d_dxSigmoid(out) * matxWeights[2,1]

    out = (matxOutputs[0,0] * matxWeights[0,0] ) + vecBais[0] + (matxOutputs[0,1] * matxWeights[0,1]) 
    vex = np.zeros(2)

    vex[0] = d_dh0* d_dxSigmoid(out)

    out = (matxOutputs[0,0] * matxWeights[1,0] ) + vecBais[1] + (matxOutputs[0,1] * matxWeights[1,1]) 
    vex[1] = d_dh1* d_dxSigmoid(out) 
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

