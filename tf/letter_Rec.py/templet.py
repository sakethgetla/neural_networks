import pygame
import math
import numpy as np

pygame.init()
gray = (115, 115, 115)
black = (0, 0, 0)
red = (255, 0, 0)
blue = (0, 0, 255)
green = (0, 255, 0)
lightGreen = (155, 255, 0)
yellow = (255, 255, 0)
white = (255, 255, 255)

display_width = 1300
display_height = 900

pointSize = 50

width = 10
height = width

clock = pygame.time.Clock()
gameDisplay = pygame.display.set_mode((display_width, display_height))
FPS = 30


class Btn():
    def __init__(self, pos, size):
        self.pos = pos
        self.size = size
        self.click= False

    def draw(self, gameDisplay):
        if(self.click):
            pygame.draw.rect(gameDisplay, black, [self.pos[0], self.pos[1], self.size[0], self.size[1]])
        else :
            pygame.draw.rect(gameDisplay, white, [self.pos[0], self.pos[1], self.size[0], self.size[1]]) 

    def clicked(self, clickPos):
        if(not self.click):
            if(clickPos[0] < self.pos[0] + self.size[0] and clickPos[0] > self.pos[0]):
                if(clickPos[1] < self.pos[1] + self.size[1] and clickPos[1] > self.pos[1]):
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

def gameLoop():
    mouse = pygame.mouse.get_pos()
    click = pygame.mouse.get_pressed()
    gameDisplay.fill(gray)
    gameExit = False
    global gravityON
    nodes = []
    notepad = []
    notepadSize = 28
    btnSize = [display_width/(3*notepadSize), display_height/(3*notepadSize)]

    for i in range(notepadSize):
        for j in range(notepadSize):
            notepad.append(Btn([int(i*btnSize[0]), int(j*btnSize[1])], btnSize))

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

        clock.tick(FPS)
        key = pygame.key.get_pressed()
        if key[pygame.K_q] :
            gameExit = True

    pygame.quit()
    quit()
gameLoop()
