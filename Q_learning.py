import numpy as np
import NN
import Snake_Q
import time
import pygame
import random

RGB = {
    0: (255, 255, 255),  # 0 = white
    1: (0, 0, 0),        # 1 = black
    2: (255, 0, 0),       # 2 = red
    3: (0, 0, 255)       # 3 = blue
}


def get_RGB(grid, _row, _column):
        return RGB[grid[_row][_column]]



    


def get_input( Snake , SnakeFieldSizeX, SnakeFieldSizeY):
    Input = np.zeros(6)
    xHead = Snake.snake[0][0]
    yHead = Snake.snake[0][1]
    for index in range(np.shape(Snake.snake)[0]):
        part = Snake.snake[index]
        #print(index)
        if index == 0:
            MinLeft  = xHead
            MinUp    = yHead
            MinRight = SnakeFieldSizeX-xHead
            MinDown  = SnakeFieldSizeY-yHead
        else:
            if xHead == part[0]:
                if part[1]-yHead < 0:
                    MinUp = min([MinUp , np.absolute(part[1]-yHead)])
                else:
                    MinDown = min([MinDown , np.absolute(part[1]-yHead)])

            elif yHead == part[1]:
                if part[0]-xHead < 0:
                    MinLeft = min([MinLeft , np.absolute(xHead-part[0])])
                else:
                    MinRight = min([MinRight , np.absolute(xHead-part[0])])
        #print(MinLeft,MinUp,MinRight,MinDown)


        Input[0] = MinLeft
        Input[1] = MinUp
        Input[2] = MinRight
        Input[3] = MinDown
        Input[4] = np.absolute(Snake.snake[0][0]-Snake.food[0])
        Input[5] = np.absolute(Snake.snake[0][1]-Snake.food[1])
        return Input



class QLearning(object):
    def Q(self, state,action,next_state,reward):
        Output = self.Network.run_NN(next_state)
        self.Q = (1-self.learning_rate) + self.learning_rate * (reward+ self.discount *np.max(max(Output)))
    def __init__(self, discount, learning_rate, epsilon_start, epsilon, epsilon_increase, memory_size, batch_size, SnakeFieldSizeX , SnakeFieldSizeY):
        self.discount         = discount
        self.learning_rate    = learning_rate                                   # alpha
        self.epsilon          = epsilon
        self.epsilon_max      = epsilon_max
        self.epsilon_increase = epsilon_increase
        self.memory_size      = memory_size
        self.batch_size       = batch_size

        self.SnakeFieldSizeX  = SnakeFieldSizeX
        self.SnakeFieldSizeY  = SnakeFieldSizeY

    def initialize_NN(self, n_inputs, neuronCounts, activation_functions):
        self.Network = NN.NN(n_inputs, neuronCounts, activation_functions)





    def run(self, runs):
        Training_set = {"Old State" : [], "Current State": [], "reward": [], "action": []}
        run_time = 0
        TS_index = 0
        Snake = Snake_Q.SnakeQ(self.SnakeFieldSizeX,self.SnakeFieldSizeY)
        Score = []
        
        while run_time < runs:
            Snake.start()
            
            while Snake.alive:
<<<<<<< HEAD
                # Get Input for the NN
                Input = get_input(Snake,self.SnakeFieldSizeX, self.SnakeFieldSizeY )
                Training_set["Old State"].append(Input)
                # Run NN
                #print('Input',Input)
                Output = self.Network.run(Input)
                #print('Output',Output)
                # Get direction (highest output value, 1 neuron left, second neuron up,
                # third neuron right, fourth neuron down)
                key = np.where(Output== np.max(max(Output)))[0][0]
=======
                if random.random() < self.epsilon:
                    # Get Input for the NN
                    Input = get_input(Snake,self.SnakeFieldSizeX, self.SnakeFieldSizeY )
                    Training_set["Old State"].append(Input)
                    # Run NN
                    #print('Input',Input)
                    Output = self.Network.run(Input)
                    #print('Output',Output)
                    # Get direction (highest output value, 1 neuron left, second neuron up,
                    # third neuron right, fourth neuron down)
                    key = np.where(Output== np.max(max(Output)))[0][0]
                else:
                    key = random.randint(0,3)
>>>>>>> 9ea4acb65cc1287855332de51da28b9bf777091c

                # Move snake in the direction
                Snake.move(self.Network, key)
                if Snake.alive and Snake.score_old < Snake.score:
                    # Reward is 1 if our score increased
                    r = 1
                elif Snake.alive:
                    # nothing happens score is decreased for punishing long ways
                    r = -0.1
                elif Snake.alive== False:
                    # Score decreased if the snake died
                    r = -10


                Input = get_input(Snake,self.SnakeFieldSizeX, self.SnakeFieldSizeY )
                Training_set["Current State"].append(Input)
                Training_set["reward"].append(r)
                Training_set["action"].append(key)
                if TS_index == memory_size:
                    TS_index = 0

                if TS_index == batch_size:
                    self.train(Training_set)

                TS_index += 1
                
            run_time += 1

            self.Network.save_NN('test')

            Score.append(Snake.score)
           
            print(Snake.score)
            print(np.mean(Score))             
    #def train(self, TS):
        

SnakeFieldSizeX  = 60
SnakeFieldSizeY  = 60
runs             = 1
discount         = 0.9
learning_rate    = 1
epsilon_start    = 0.3
epsilon_max      = 0.9
epsilon_increase = 0.1
memory_size      = 500
batch_size       = 400

a = QLearning(discount, learning_rate, epsilon_start, epsilon_max, epsilon_increase, memory_size, batch_size, SnakeFieldSizeX, SnakeFieldSizeY)
a.initialize_NN(6, [15,15,4], ['sigmoid','sigmoid','sigmoid'])
#Snake = Snake_Q.SnakeQ(60,60)
#Snake.start()
#Snake.move(a.Network)

a.run(runs)
#a.Network.printNN()
