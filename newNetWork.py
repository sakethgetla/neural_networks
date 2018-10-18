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

matxWeights = np.array([[0, 0], [0, 0], [0]])


print("matxWeights " + str(matxWeights))

print(matxWeights[2] + 2)

#im = cv2.imread("aaa.png", mode='RGB')
#print(type(im))

img = mpimg.imread('aaa.png')
print(type(img))
#print(img)

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


gameLoop()
