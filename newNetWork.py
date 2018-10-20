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
stepSize = 1


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

training = AND
print("training ")
print(training)

gameDisplay = pygame.display.set_mode((display_width, display_height))

pygame.display.set_caption('Nnet Visual')

font = pygame.font.SysFont(None, 25)


#img = mpimg.imread('aaa.png')
#print(type(img))
##print(img)

matxWeights = np.random.randint(6,size=(2,3))
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

vecBais = np.array((1,1,1))
print(" ==")
print(vecBais)


def calculateOutput(input,w,b):
    #w #weights 2d vector
    #b #bais
    #input #training set 2d vector

    print("b " + str(b))
    
    print("weight ")
    print(w)

    print("input ")
    print(input)
    
    print("a = np.multiply(np.identity(2),input)")
    a = np.multiply(np.identity(2),input)
    print(a)

    print("input transpose ")
    input = np.transpose(input)
    print(input)

    print("a = np.multiply(np.identity(2),input)")
    a = np.multiply(np.identity(2),input)
    print(a)

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
    print("np.multiply(multiply,np.ones(2))")
    print(np.multiply(multiply,np.ones(2)))

    print(multiply)
    print(multiply[0])
    print(multiply[1])
    sa = np.vectorize(multiply)
    print("np.diag(multiply)")
    multiply = np.diag(multiply)
    print(multiply)
    
    print("addBias = np.add(multiply,b)")
    addBias = np.add(multiply,b)
    print(addBias)
    print(np.transpose( addBias))
    print("ddddddddddd")
    print( np.reshape( addBias,(-1,1)))
#    matxOutputs[:,0] =  addBias
 #   print("matrix output" + str(matxOutputs))
    return   addBias

#    matxOutputs[:,0:1] = np.reshape( addBias,(-1,1))
#    print(matxOutputs)


    #matxOutputs[:,0:1] = np.multiply(identy,input)
        

def sigmoid(input):
    output = 1/(1+math.exp(-input))
    print("sigmoid output " + str(output))


def update(gameDisplay, color):
    myfont = pygame.font.SysFont("monospace", 15)
    output = self.output
    label = myfont.render(str(output), 1, (0, 255, 255))
    gameDisplay.blit(label, self.XYpos)
    pygame.draw.circle(gamedisplay, color, self.XYpos, 20, 3)


def gameLoop():
    gameDisplay.fill(gray)
    currPlace = 0
    gameExit = False
    while not gameExit:
        pygame.display.update()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                gameExit = True
        clock.tick(FPS)
        key = pygame.key.get_pressed()
        if key[pygame.K_d]:
            print("corrpos " + str(currPlace))
            if(currPlace % 4 == 3):
                currPlace += 1
                calResult(training[0][0])
            elif(currPlace % 4 == 2):
                currPlace += 1
                calResult(training[1][0]) 
            elif(currPlace % 4 == 1):
                currPlace += 1
                calResult(training[2][0])
            elif(currPlace % 4 == 0):
                currPlace += 1
                calResult(training[3][0])
    pygame.quit()
    quit()


#i=0
#print("train")
#print(training[0][i])
#print("input weights")
#print(matxWeights[:,i:1])
#calculateOutput([training[0][i]],matxWeights[:,i:1],vecBais[i])
#
def calResult(train):
    for p in range(4):
        print("train")
        print(train)
        print("input weights")
        print(matxWeights[:,0])
        print(calculateOutput([train],matxWeights[:,0],vecBais[0]))
        matxOutputs[:,0] = calculateOutput([train],matxWeights[:,0], vecBais[0])
        print("matrix output" + str(matxOutputs))
        matxOutputs[:,1] = calculateOutput([matxOutputs[:,0]],matxWeights[:,1],vecBais[1])
        print("matrix output" + str(matxOutputs))                                   
        matxOutputs[:,2] = calculateOutput([matxOutputs[:,1]],matxWeights[:,2],vecBais[2])
        print(matxWeights)
        print("matrix output")
        print(matxOutputs)




gameLoop()
