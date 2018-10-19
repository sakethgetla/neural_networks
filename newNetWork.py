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
    print("return  np.multiply(identyWeight,input)")
    return  np.multiply(identyWeight,input)
    #matxOutputs[:,0:1] = np.multiply(identy,input)
    
    

def sigmoid(self, input):
    self.output = 1/(1+math.exp(-input))
    print("sigmoid output " + str(self.output))


def update(self, gameDisplay, color):
    myfont = pygame.font.SysFont("monospace", 15)
    output = self.output
    label = myfont.render(str(output), 1, (0, 255, 255))
    gameDisplay.blit(label, self.XYpos)
    pygame.draw.circle(gameDisplay, color, self.XYpos, 20, 3)


def gameLoop():
    gameDisplay.fill(gray)
    gameExit = False
    while not gameExit:
        pygame.display.update()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                gameExit = True
        clock.tick(FPS)
    pygame.quit()
    quit()

print(calculateOutput([[2,3]],matxWeights[:,0:1],vecBais))
gameLoop()
