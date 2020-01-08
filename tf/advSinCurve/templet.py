import pygame
import math
import numpy as np
from visualSinCurve import initalise, runEpoch
#from visualSinCurve import initalise, runEpoch, MyModel

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
FPS = 5
x_range = (0, math.pi*6)
y_range = (-15, 15)

def draw(predictions, test_ds):
    x_test, y_test = test_ds
    for x, y, z in zip(x_test, y_test, predictions):
        #for x, y, z in zip(test_ds, predictions):
        x1 = x*display_width / x_range[1]
        y1 = y*display_height / (y_range[1]*2)
        z1 = z*display_height / (y_range[1]*2)
        pygame.draw.circle(gameDisplay, red, [x1, y1], 4)
        pygame.draw.circle(gameDisplay, red, [z1, y1], 4)

def gameLoop():
    mouse = pygame.mouse.get_pos()
    click = pygame.mouse.get_pressed()
    gameDisplay.fill(gray)
    gameExit = False
    global gravityON
    nodes = []
    train_ds, test_ds, model, loss_obj, optimizer = initalise()

    EPOCHS = 100

    while not gameExit:
        mouse = pygame.mouse.get_pos()
        click = pygame.mouse.get_pressed()

        if (EPOCHS > 0):
            EPOCHS -= 1
            predictions, model, loss_obj, optimizer = runEpoch(train_ds, test_ds, model, loss_obj, optimizer)

        draw(predictions, train_ds)

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
