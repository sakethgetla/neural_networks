import pygame
import math
import numpy as np
import tensorflow as tf
from pdb import set_trace as bp

from tensorflow.keras.layers import Dense, InputLayer, Flatten, Conv2D
#from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

#from learnNums.py import MyModel

pygame.init()
gray = (115, 115, 115)
black = (0, 0, 0)
red = (255, 0, 0)
blue = (0, 0, 255)
green = (0, 255, 0)
lightGreen = (155, 255, 0)
yellow = (255, 255, 0)
white = (255, 255, 255)

#display_width = 800
#display_height = 800
display_width = 28*7
display_height = 28*7

pointSize = 50

width = 10
height = width

clock = pygame.time.Clock()
gameDisplay = pygame.display.set_mode((display_width, display_height))
FPS = 30
ans = np.zeros((1, 28, 28, 1))

#model = MyModel()
#model.load_weights('weights')

def dist(x1, x2):
    return math.sqrt((x1[0] - x2[0])**2 + (x1[1] - x2[1])**2 )

class Btn():
    def __init__(self, pos, size, gridPos):
        self.pos = pos
        self.size = size
        self.click= False
        self.gridPos= gridPos

    def draw(self, gameDisplay):
        if(self.click):
            pygame.draw.rect(gameDisplay, black, [self.pos[0], self.pos[1], self.size[0], self.size[1]])
        else :
            pygame.draw.rect(gameDisplay, white, [self.pos[0], self.pos[1], self.size[0], self.size[1]]) 

    def clicked(self, clickPos):
        if(not self.click):
            #if (dist(clickPos, self.pos) < 10):
            if (dist(clickPos, (self.pos[0]+(self.size[0]/2), self.pos[1]+(self.size[1]/2))) < self.size[0]*2):
            #if(clickPos[0] < self.pos[0] + self.size[0] and clickPos[0] > self.pos[0]):
            #    if(clickPos[1] < self.pos[1] + self.size[1] and clickPos[1] > self.pos[1]):
                ans[0][self.gridPos[0], self.gridPos[1]] = 1
                self.click = True



#def btn(gameDisplay, pos, size,  color, txt, action, mouse, click):
#    global startBtnClicked
#    #click = pygame.mouse.get_rel()
#    #print(click)
#    click = pygame.mouse.get_pressed()
#
#    if (click[0] == 1 and action != None ):
#        if ( pos[0] < mouse[0] < pos[0]+size[0] and not startBtnClicked):
#            if ( pos[1] < mouse[1] < pos[1]+size[1]):
#                startBtnClicked = True
#                action()
#    else :
#        startBtnClicked = False
#
#    pygame.draw.rect(gameDisplay, color, [pos[0], pos[1], size[0], size[1]] )
#    myfont = pygame.font.SysFont("monospace", 15)
#    label = myfont.render(str(txt), 1, black)
#    gameDisplay.blit(label, pos)

def draw():
    pass

class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        #self.input = InputLayer(input_tensor=np.shape(x_train[0]))
        #self.inp= InputLayer()
        #self.inp= InputLayer(input_tensor=x_train[0])
        #self.d1 = Dense(5, activation='sigmoid')
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='tanh')
        #self.d1 = Dense(15, activation='tanh', input_shape = [28, 28, 1])
        #self.d1 = Dense(10,  activation='relu')
        self.d2 = Dense(10, activation='softmax')
        #self.d3 = Dense(1, activation='tanh')
        #self.d3 = Dense(1, activation='sigmoid')
        #self.d3 = Dense(1, activation='sigmoid', use_bias=False)
        #self.d3 = Dense(1)
        #self.d4 = Dense(1, activation='relu')

    def call(self, x):
        #x = self.inp(x)
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        #x = self.d2(x)
        #x = self.d3(x)
        #x = self.d4(x)
        return self.d2(x)

    #def __init__(self):
    #    super(MyModel, self).__init__()
    #    #self.input = InputLayer(input_tensor=np.shape(x_train[0]))
    #    #self.inp= InputLayer()
    #    #self.inp= InputLayer(input_tensor=x_train[0])
    #    #self.d1 = Dense(5, activation='sigmoid')
    #    self.conv1 = Conv2D(32, 3, activation='relu')
    #    self.flatten = Flatten()
    #    self.d1 = Dense(100, activation='tanh')
    #    #self.d1 = Dense(15, activation='tanh', input_shape = [28, 28, 1])
    #    #self.d1 = Dense(10,  activation='relu')
    #    self.d2 = Dense(20, activation='tanh')
    #    #self.d3 = Dense(1, activation='tanh')
    #    #self.d3 = Dense(1, activation='sigmoid')
    #    #self.d3 = Dense(1, activation='sigmoid', use_bias=False)
    #    self.d3 = Dense(1)
    #    #self.d4 = Dense(1, activation='relu')

    #def call(self, x):
    #    #x = self.inp(x)
    #    x = self.conv1(x)
    #    x = self.flatten(x)
    #    x = self.d1(x)
    #    x = self.d2(x)
    #    #x = self.d3(x)
    #    #x = self.d4(x)
    #    return self.d3(x)


model = MyModel()
#bp()
model.predict(ans)
model.load_weights('weights')

def gameLoop():

    mouse = pygame.mouse.get_pos()
    click = pygame.mouse.get_pressed()
    gameDisplay.fill(gray)
    gameExit = False
    global gravityON
    nodes = []
    notepad = []
    notepadSize = 28
    #btnSize = [display_width/(3*notepadSize), display_height/(3*notepadSize)]
    btnSize = [display_width/28, display_height/28]

    for i in range(notepadSize):
        for j in range(notepadSize):
            notepad.append(Btn([int(i*btnSize[0]), int(j*btnSize[1])], btnSize, (i, j)))

    while not gameExit:
        mouse = pygame.mouse.get_pos()
        click = pygame.mouse.get_pressed()

        for tile in notepad:
            if( click[0] == 1):
            #if (click[0] == 1 and action != None ):
                tile.clicked([int(mouse[0]), int(mouse[1])])
            tile.draw(gameDisplay)

        pygame.display.update()
        gameDisplay.fill(gray)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                gameExit = True

        print(model(ans))
        clock.tick(FPS)
        key = pygame.key.get_pressed()
        if key[pygame.K_q] :
            gameExit = True

    pygame.quit()
    quit()
gameLoop()
