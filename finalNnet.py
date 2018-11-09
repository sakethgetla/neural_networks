import matplotlib.image as mpimg
import pygame
import math
import numpy as np
 
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

FPS = 10
speed = 5
dist = 150
stepSize = 0.1
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

for i in range(2):
    neuronPos.append([dist*(3) , dist*(i+1)])

training = XOR 
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
matxWeights += np.random.randint(9,size=(3,2))
print("matxWeights ==")
print(matxWeights)

matxOutputs = np.zeros((3,2))
print("matxOutputs ==")
print(matxOutputs)

vecBais = np.array((1,1))
print(" vecBais")
print(vecBais)

matxDs = np.zeros((3,2))
matxRnd = np.random.randint(9,size=(3,2))
print("matxDs")
print(matxDs)

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
    global matxWeights
    global matxDs 

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
        if key[pygame.K_d]:
            print("training " + str(training))
            global error 
            error =0 

            run(training[0])
            run(training[1])
            run(training[2])
            run(training[3])
            matxWeights += matxDs * stepSize *(1)
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


gameLoop()
