from pdb import set_trace as bp
import pygame
import math
import numpy as np
from visualSinCurve import initalise, train_step
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
#bp()
gameDisplay = pygame.display.set_mode((display_width, display_height))
FPS = 5
x_range = (0, math.pi*6)
y_range = (-15, 15)
#bp()
 
EPOCHS = 100


def draw(predictions, x_test, y_test):
    counter = 0
    #for x, y in (x_test, y_test):
    #for counter in range(x_test):
    for x, y in zip(x_test, y_test):
        #print(type(x_test[1]))
        #bp()
        x1 = x[1]*display_width / x_range[1]
        y1 =( y*display_height / (y_range[1]*2)) + (display_height//2)

        #print(f"counter = {counter}, x = {x}, y = {y}, predictions = {predictions[counter]}, ")
        if (predictions[counter] > 20 and predictions[counter] < -20):
            z1 = y_range[1]
        else:
            z1 = (predictions[counter]*display_height / (y_range[1]*2)) + (display_height//2)
        #print(f"x1 = {x1}, y1 = {y1}, z1 = {z1}, ")
        counter +=1

        #if (z1 > 0 and z1 < (display_height/2)):
        pygame.draw.circle(gameDisplay, red, [int(x1), int(z1)], 4)
        pygame.draw.circle(gameDisplay, blue, [int(x1), int(y1)], 4)

    #for x, y, z in zip(x_test, y_test, predictions):
    #    #for x, y, z in zip(test_ds, predictions):
    #    x1 = x*display_width / x_range[1]
    #    y1 = y*display_height / (y_range[1]*2)
    #    z1 = z*display_height / (y_range[1]*2)
    #    pygame.draw.circle(gameDisplay, red, [x1, y1], 4)
    #    pygame.draw.circle(gameDisplay, red, [z1, y1], 4)

def gameLoop():
    #bp()
    mouse = pygame.mouse.get_pos()
    click = pygame.mouse.get_pressed()
    gameDisplay.fill(gray)
    gameExit = False
    global gravityON
    nodes = []
    train_ds, test_ds, model, loss_obj, optimizer = initalise()

    EPOCHS = 500

    while not gameExit:
        mouse = pygame.mouse.get_pos()
        click = pygame.mouse.get_pressed()

        if (EPOCHS > 0):
            print(f"{EPOCHS} EPOCHS to go")
            EPOCHS -= 1
            #predictions, model, loss_obj, optimizer = runEpoch(train_ds, test_ds, model, loss_obj, optimizer)
            #for epoch in range(EPOCHS):
            # Reset the metrics at the start of the next epoch
            #train_loss.reset_states()
            #train_accuracy.reset_states()
            #test_loss.reset_states()
            #test_accuracy.reset_states()

            #for x, y in zip(x_train, y_train):
            #model.predict(x_train[:2])
            for x, y in train_ds:
                train_step(x, y, model, loss_obj, optimizer)
                #model = train_step(x, y, model, loss_obj, optimizer)

            #for x, y in zip(x_test, y_test):
            x_test = test_ds[['x0', 'x1']].values
            y_test = test_ds['y'].values
            #bp()
            predictions = model(x_test)
            #model_hist.append(train_loss.result())

        draw(predictions, x_test, y_test)

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
