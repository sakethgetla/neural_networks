import pygame
import time
import random

pygame.init()

white =   (255,255,255)
black = (0,0,0)
red = (255,0,0)
green = (0,155,0)

display_width = 800
display_height  = 600

clock = pygame.time.Clock()

FPS = 30
speed = 5
gameDisplay = pygame.display.set_mode((display_width,display_height))
pygame.display.set_caption('Nnet Visual')

font = pygame.font.SysFont(None, 25)


def gameLoop():
    print("awdkj")
    gameExit = False
    x = 10
    y = 10
    while not gameExit:
        print(x)
        gameDisplay.fill(white)
        pygame.draw.circle(gameDisplay, red, [x,y],50)
        clock.tick(FPS)
        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                gameExit = True
            if event.type == pygame.key.get_pressed:
                if event.key[K_LEFT]:
                    x -= speed
                elif event.key == pygame.K_RIGHT:
                    x += speed
                elif event.key == pygame.K_UP:
                    y -= speed
                elif event.key == pygame.K_DOWN:
                    y += speed
                    
gameLoop()
