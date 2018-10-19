import pygame
import math
import numpy as np
#import matplotlib.pyplot as plt

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

#empty = np.zeros((5,5))
#print("zeros " + str(empty))
#p = empty[0:1]
#print("sefse" +p)
#

#identy = np.identity(5,None)
#print(identy)
#
#p = identy[0:3]
p
#identy = np.matrix((list[4],list[5]))
#print(identy)
#
#p = identy[0:3]
#print("sefse" + str(p))
#
#w= matxWeights[2] + 2
#
#print(w)
#print(matxWeights[2] + 2)
#
#im = cv2.imread("aaa.png", mode='RGB')
#print(type(im))



FPS = 30
speed = 5
dist = 150
stepSize = 1


inputNeurons = []
hiddenNeurons = []
outputNeurons = []


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


# def train(gameDisplay = pygame.display.set_mode((display_width,display_height))
gameDisplay = pygame.display.set_mode((display_width, display_height))

pygame.display.set_caption('Nnet Visual')

font = pygame.font.SysFont(None, 25)

a_list = ['dwad', 'Awdaw', 'dawd']


def f(t):
    return 1/(1+np.exp(-t))


# t2 = np.arange(-5.0, 5.0, 0.02)
# plt.subplot(111)
# plt.plot(t2, f(t2), 'k')
# plt.show()
#####################################

# plt.subplot(212)
# plt.plot(t2, np.cos(2*np.pi*t2), 'r--')


# ax = plt.subplot(111)
#
# t = np.arange(0.0, 5.0, 0.01)
# s = 1/(1+math.exp(-t))
# line, = plt.plot(t, s, lw=2)
#
# plt.annotate('local max', xy=(2, 1), xytext=(3, 1.5),
#            arrowprops=dict(facecolor='black', shrink=0.05),
#            )
#
# plt.ylim(-2,2)
# plt.show()


class Neuron():
    XYpos = [1, 2]
    weights = [1, 1]
    output = 0

    def __init__(self, gamedisplay, xy, color):
        self.XYpos = xy
        pygame.draw.circle(gameDisplay, color, self.XYpos, 20)

    def connect(self, gameDisplay, XY1, XY2, color):
        self.weights
        pygame.draw.line(gameDisplay, color, XY1, self.XYpos, self.weights[0])
        pygame.draw.line(gameDisplay, color, XY2, self.XYpos, self.weights[1])

    def sigmoid(self, input):
        self.output = 1/(1+math.exp(-input))
        print("sigmoid output " + str(self.output))

    def update(self, gameDisplay, color):
        myfont = pygame.font.SysFont("monospace", 15)
        output = self.output
        label = myfont.render(str(output), 1, (0, 255, 255))
        gameDisplay.blit(label, self.XYpos)
        pygame.draw.circle(gameDisplay, color, self.XYpos, 20, 3)

# def rect(x,y,width,height):


def calculateOutput(input):
    i = 0
    print("cal input " + str(input))
    for n in inputNeurons:
        print("inputNeurons: " + str(i) + " input= " + str(input[0][i]))
        n.sigmoid(input[0][i])
        i += 1

    i = 0
    for n in hiddenNeurons:
        add = inputNeurons[0].output * inputNeurons[0].weights[i]
        add += inputNeurons[1].output * inputNeurons[1].weights[i]
        print("hiddenNeurons: " + str(i) + " input= " + str(add))
        n.sigmoid(add)
        i += 1

    i = 0
    for n in outputNeurons:
        add = hiddenNeurons[0].output * hiddenNeurons[0].weights[i]
        add += hiddenNeurons[1].output * hiddenNeurons[1].weights[i]
        print("outputNeurons: " + str(i) + " input= " + str(add))
        n.sigmoid(add)
        i += 1


def calError(input):
    if(input[1] == 0):
        totError = absolDist(outputNeurons[0].output, 1) + absolDist(outputNeurons[1].output, 0)
    elif(input[1] == 1):
        totError = absolDist(outputNeurons[0].output, 0) + absolDist(outputNeurons[1].output, 1)
    return totError


def absolDist(yExpexted, yOutput):
    return ((yExpexted - yOutput)**2)/2


def d_dxSigmoid(input):
    return ((1 + math.exp(-input))**(-2))*(math.exp(-input))


def gameLoop():
    #    print(training)
    gameDisplay.fill(gray)
    gameExit = False

#    connXY = [[0, 0], [0, 0]]
    i = 0
    XY = [dist*(i+1), dist]
    x = Neuron(gameDisplay, XY, green)
    inputNeurons.append(x)

    XY = [dist*(i+1), dist*2]
    x = Neuron(gameDisplay, XY, green)
    inputNeurons.append(x)

    i = 1
    XY = [dist*(i+1), dist-50]
    x = Neuron(gameDisplay, XY, green)
    hiddenNeurons.append(x)

    XY = [dist*(i+1), dist*2+50]
    x = Neuron(gameDisplay, XY, green)
    hiddenNeurons.append(x)

    i = 2
    XY = [dist*(i+1), dist]
    x = Neuron(gameDisplay, XY, green)
    outputNeurons.append(x)

    XY = [dist*(i+1), dist*2]
    x = Neuron(gameDisplay, XY, green)
    outputNeurons.append(x)

    matxWeights = [inputNeurons[0].weights,
                   inputNeurons[1].weights,
                   hiddenNeurons[0].weights,
                   hiddenNeurons[1].weights,
                   outputNeurons[0].weights,
                   outputNeurons[1].weights]

    #    matxWeights = np.matrix([inputNeurons[0].weights,
#                             inputNeurons[1].weights,
#                            hiddenNeurons[0].weights,
#                            hiddenNeurons[1].weights,
#                            outputNeurons[0].weights,
#                            outputNeurons[1].weights])
    print(matxWeights)
    currPlace = 0
#    calculateOutput(training[0][0])
#    for n in inputNeurons:
#        n.connect(gameDisplay,hiddenNeurons[0].XYpos,hiddenNeurons[1].XYpos,red)
    while not gameExit:
        gameDisplay.fill(gray)
        update(matxWeights)
        pygame.display.update()
        key = pygame.key.get_pressed()
        # checking pressed keys
        if key[pygame.K_d]:
            print("corrpos " + str(currPlace))
            if(currPlace % 4 == 3):
                currPlace += 1
                calculateOutput(training[0])
            elif(currPlace % 4 == 2):
                currPlace += 1
                calculateOutput(training[1])
            elif(currPlace % 4 == 1):
                currPlace += 1
                calculateOutput(training[2])
            elif(currPlace % 4 == 0):
                currPlace += 1
                calculateOutput(training[3])
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                gameExit = True
        clock.tick(FPS)
    pygame.quit()
    quit()


#    1: print(calculateOutput(training[1]))
#    2:
#        print(calculateOutput(training[2]))
#        print(calculateOutput(training[3]))


def update(matxWeights):
    inputNeurons[0].weights = matxWeights[0]
    inputNeurons[1].weights = matxWeights[1]
    hiddenNeurons[0].weights = matxWeights[2]
    hiddenNeurons[1].weights = matxWeights[3]
    outputNeurons[0].weights = matxWeights[4]
    outputNeurons[1].weights = matxWeights[5]

    for n in inputNeurons:
        n.update(gameDisplay, green)
        n.connect(gameDisplay, hiddenNeurons[0].XYpos, hiddenNeurons[1].XYpos, red)

    for n in hiddenNeurons:
        n.update(gameDisplay, green)
        n.connect(gameDisplay, outputNeurons[0].XYpos, outputNeurons[1].XYpos, red)

    for n in outputNeurons:
        n.update(gameDisplay, green)


gameLoop()
