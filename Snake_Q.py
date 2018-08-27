from random import randint
import numpy as np




def leftStart(WinSizeX,WinSizeY):

    xHead = randint(5,WinSizeX-7)
    yHead = randint(5,WinSizeY-7)
    snake = [[xHead,yHead], [xHead,yHead+1], [xHead,yHead+2]]                                     # Initial snake coordinates
    
    food = [randint(1, WinSizeX-2), randint(1, WinSizeY-2)]                                                       # Initial food coordinates
    while food in snake:
        food = [randint(1, WinSizeX-2), randint(1, WinSizeY-2)]                                                             # Draw food
    key = 0                                                                               # Start direction snake
    return food,snake,key

def upStart(WinSizeX,WinSizeY):
    xHead = randint(5,WinSizeX-7)
    yHead = randint(5,WinSizeY-7)
    snake = [[xHead,yHead], [xHead+1,yHead], [xHead+2,yHead]]                                     # Initial snake coordinates
    
    food = [randint(1, WinSizeX-2), randint(1, WinSizeY-2)]                                                       # Initial food coordinates
    while food in snake:
        food = [randint(1, WinSizeX-2), randint(1, WinSizeY-2)]                                                             # Draw food
    key = 1                                                                                 # Start direction snake
    return food,snake,key

def rightStart(WinSizeX,WinSizeY):
    xHead = randint(5,WinSizeX-7)
    yHead = randint(5,WinSizeY-7)
    snake = [[xHead,yHead], [xHead,yHead-1], [xHead,yHead-2]]                                     # Initial snake coordinates
    
    food = [randint(1, WinSizeX-2), randint(1, WinSizeY-2)]                                                       # Initial food coordinates
    while food in snake:
        food = [randint(1, WinSizeX-2), randint(1, WinSizeY-2)]                                                            # Draw food
    key = 2                                                                             # Start direction snake
    return food,snake,key

def downStart(WinSizeX,WinSizeY):
    xHead = randint(5,WinSizeX-7)
    yHead = randint(5,WinSizeY-7)
    snake = [[xHead,yHead], [xHead-1,yHead], [xHead-2,yHead]]                                     # Initial snake coordinates
    
    food = [randint(1, WinSizeX-2), randint(1, WinSizeY-2)]                                                       # Initial food coordinates
    while food in snake:
        food = [randint(1, WinSizeX-2), randint(1, WinSizeY-2)]                                                             # Draw food
    key = 3                                                                                # Start direction snake
    return food,snake,key


def switch_start(argument,WinSizeX,WinSizeY):
    switcher = {
        1: rightStart,
        2: downStart,
        3: leftStart,
        4: upStart,
    }
    
    startIni = switcher.get((argument))
    food,snake,key = startIni(WinSizeX,WinSizeY)
#    print('start',key,snake)
    return food,snake


   


class SnakeQ(object):
    
    def __init__(self, WinSizeX, WinSizeY):
        self.WinSizeX = WinSizeX
        self.WinSizeY = WinSizeY
        self.all_score = []
        
    def start(self):
        startDirection  = randint(1, 4)                                            # random start direction 1: right, 2: down, 3: left, 4: up
        food,snake      = switch_start(startDirection,self.WinSizeX,self.WinSizeY)
        self.food       = food
        self.snake      = snake
        self.alive      = True
        self.score      = 0
        self.score_old  = 0
        
    def move(self, Network, key):

        #print(key)
        #print(self.snake) 
        # Calculates the new coordinates of the head of the snake
        self.snake.insert(0, [self.snake[0][0] + (key == 3 and 1) + (key == 1 and -1), self.snake[0][1] + (key == 0 and -1) + (key == 2 and 1)])
        
        # If snake crosses the boundaries kill it
        if self.snake[0][0] == 0: self.alive = False 
        if self.snake[0][1] == 0: self.alive = False 
        if self.snake[0][0] == self.WinSizeX-1: self.alive = False 
        if self.snake[0][1] == self.WinSizeY-1: self.alive = False 
        # If snake runs into itself kill it
        if self.snake[0] in self.snake[1:]: self.alive = False
        self.score_old = self.score  
        # When snake eats the food             
        if self.snake[0] == self.food:
            self.food = []                                        
            self.score += 1
            while self.food == []:
                self.food = [randint(1, self.WinSizeX-2), randint(1, self.WinSizeY-2)]                 # Calculating next food's coordinates
                if self.food in self.snake: self.food = []
        else:    
            last = self.snake.pop() 
                                        # If it does not eat the food, length decreases

        
        
        
        
        
        
        
        
        
        
        
        