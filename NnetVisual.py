import pygame
import time
import random

pygame.init()

white = (255,255,255)

black = (0,0,0)
red = (255,0,0)
green = (0, 155, 0)

display_width = 800
display_height  = 600

width = 50
height =width

clock = pygame.time.Clock()

FPS = 30
speed = 5
dist = 150

inputNeurons=[]
hiddenNeurons=[]
outputNeurons=[]


XOR = [
    [[0,0],[0]],
    [[0,1],[1]],    
    [[1,0],[1]],
    [[1,1],[0]],
]

AND = [
    [[0,0],[0]],
    [[0,1],[0]],    
    [[1,0],[0]],
    [[1,1],[1]],
]

training = AND

print( training[0][1] )


#def train(gameDisplay = pygame.display.set_mode((display_width,display_height))
gameDisplay = pygame.display.set_mode((display_width,display_height))

pygame.display.set_caption('Nnet Visual')

font = pygame.font.SysFont(None, 25)

a_list = ['dwad', 'Awdaw' ,'dawd']




class Neuron():
    pos=[]
    weights =[1,1]
    output = 0
        
    def __init__(self,gamedisplay,xy,color):
        pos = xy
        pygame.draw.circle(gameDisplay, color, pos,20)

    def connect(gameDisplay,XY1,XY2, color, widths):
        weights = widths
#        pygame.draw.rect(gameDisplay, green, [x,y,width,width])
#        pygame.draw.polygon(gameDisplay, green, [x,y,width,width])
        pygame.draw.line(gameDisplay,color,XY1[0],XY1[1],weights[0])
        pygame.draw.line(gameDisplay,color,XY2[0],XY2[1],weights[1])

    def sigmoid(input):
        output = 1/(1+math.exp(-input))


#def rect(x,y,width,height):
    
def calculateOutput(input):

    i =0

    for n in inputNeurons:
        n.sigmoid(input[i])
        i++

    i=0
    for n in hiddenNeurons:
        add = inputNeurons[0].output * inputNeurons[0].weights[i] 
        add +=inputNeurons[1].output * inputNeurons[1].weights[i]
        n.sigmoid(add)
 
        i++
    

    i=0
    for n in outputNeurons:
        add = hiddenNeurons[0].output * hiddenNeurons[0].weights[i]
        add += hiddenNeurons[1].output * hiddenNeurons[1].weights[i]
        n.sigmoid(add)
        i++


    
    



def gameLoop():

    gameExit = False
    x = 10
    y = 10

    connXY = [[0,0],[0,0]]

            
        i =0
        XY = [dist*(i+1),dist]
        x =  Neuron(gameDisplay,XY,green)
        inputNeurons.append(x)

        
        XY = [dist*(i+1),dist*2]
        x =  Neuron(gameDisplay,XY,green)
        inputNeurons.append(x)
        
        i =1
        XY = [dist*(i+1),dist]
        x =  Neuron(gameDisplay,XY,green)
        hiddenNeurons.append(x)
        
        XY = [dist*(i+1),dist*2]
        x =  Neuron(gameDisplay,XY,green)
        hiddenNeurons.append(x)

        i =2
        XY = [dist*(i+1),dist]
        x =  Neuron(gameDisplay,XY,green)
        outputNeurons.append(x)
        
        XY = [dist*(i+1),dist*2]
        x =  Neuron(gameDisplay,XY,green)
        outputNeurons.append(x)

    
    while not gameExit:
        
        #print(x)
        gameDisplay.fill(white)

        calculateOutput()        
        
        # circle(dist*(i+1),dist)    
        # circle(dist*(i+1),dist*2)
        
        # i=1
        # circle(dist*(i+1),dist)
        # circle(dist*(i+1),dist*2)
        
        # i=2
        # circle(dist*(i+1),dist)
        # circle(dist*(i+1),dist*2)



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
    clock.tick(FPS)
    pygame.quit()
    quit()

gameLoop()
