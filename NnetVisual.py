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
dist = 150


gameDisplay = pygame.display.set_mode((display_width,display_height))
pygame.display.set_caption('Nnet Visual')

font = pygame.font.SysFont(None, 25)

def circle(x,y):
    pygame.draw.circle(gameDisplay, red, [x,y],20)
    
def rect(x,y,width,height):
    pygame.draw.rect(gameDisplay, green, [x,y,width,height])

def gameLoop():
    print("awdkj")
    gameExit = False
    x = 10
    y = 10
    while not gameExit:
        print(x)
        gameDisplay.fill(white)

        for i in range(3):
            circle(dist*(i+1),dist)
#           rect(45,21,45,45)

        for i in range(3):
            circle(dist*(i+1),dist*2)
#            rect(45,21,45,45)

        
        
        clock.tick(FPS)
        pygame.display.update()

        key=pygame.key.get_pressed()  #checking pressed keys
        if key[pygame.K_a]: x -= speed 
        if key[pygame.K_d]: x += speed
        if key[pygame.K_LEFT]: y -= speed 
        


        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                gameExit = True


#            if event.type == pygame.key.get_pressed:
#                if event.key[K_LEFT]:
#                    x -= speed
#                elif event.key == pygame.K_RIGHT:
#                    x += speed
#                elif event.key == pygame.K_UP:
#                    y -= speed
#                elif event.key == pygame.K_DOWN:
#                    y += speed
#                    
gameLoop()
