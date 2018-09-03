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
    def __init__(self, discount, learning_rate, epsilon_start, epsilon_max, epsilon_increase, memory_size, batch_size, SnakeFieldSizeX , SnakeFieldSizeY):
        self.discount         = discount
        self.learning_rate    = learning_rate                                   # alpha
        self.epsilon          = epsilon_start
        self.epsilon_max      = epsilon_max
        self.epsilon_increase = epsilon_increase
        self.memory_size      = memory_size
        self.batch_size       = batch_size

        self.SnakeFieldSizeX  = SnakeFieldSizeX
        self.SnakeFieldSizeY  = SnakeFieldSizeY

    def initialize_NN(self, n_inputs, neuronCounts, activation_functions):
        self.Network = NN.NN(n_inputs, neuronCounts, activation_functions)


    def run(self, runs):
        Training_set = []
        run_time = 0
        memory_index = 0
        batch_index = 0
        Snake = Snake_Q.SnakeQ(self.SnakeFieldSizeX,self.SnakeFieldSizeY)
        Score = []

        while run_time < runs:
            Snake.reward = 0
            Snake.start()

            while Snake.alive:
                # Get pressed keys
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            pygame.quit()
                            return
                        elif event.key == pygame.K_UP:
                            Snake.show = not Snake.show
                        elif event.key == pygame.K_p:
                            self.Network.printNN()

                if len(Training_set) <= memory_index:
                    Training_set.append({})

                # Get Input for the NN
                Input = get_input(Snake,self.SnakeFieldSizeX, self.SnakeFieldSizeY )
                Training_set[memory_index]["Old State"] = Input

                if random.random() < self.epsilon:
                    # Run NN
                    #print('Input',Input)
                    Output = self.Network.run(Input)
                    #print('Output',Output)
                    # Get direction (highest output value, first neuron left, second neuron up,
                    # third neuron right, fourth neuron down)
                    key = np.argmax(Output)
                else:
                    key = random.randint(0,3)


                # Move snake in the direction
                Snake.move(key)
                if Snake.alive and Snake.score_old < Snake.score:
                    # Reward is 1 if our score increased
                    r = 10
                elif Snake.alive:
                    # nothing happens score is decreased for punishing long ways
                    r = 0.1
                elif Snake.alive== False:
                    # Score decreased if the snake died
                    r = -10
                Snake.reward += r
                Input = get_input(Snake, self.SnakeFieldSizeX, self.SnakeFieldSizeY)
                Training_set[memory_index]["Current State"] = Input
                Training_set[memory_index]["reward"] = r
                Training_set[memory_index]["action"] = key

                memory_index += 1
                batch_index  += 1

                if memory_index == self.memory_size:
                    memory_index = 0
                if batch_index == self.batch_size:
                    batch_index = 0
                    self.train(Training_set)


            run_time += 1

            self.Network.save_NN('test')

            Score.append(Snake.score)

            print("Run:", run_time, "Score:", Snake.score, "Total reward:", Snake.reward)
            #print(np.mean(Score))
        pygame.quit()


    def train(self, TS):
        #sample_indx = np.random.choice([i for i in range(self.memory_size)], self.batch_size, replace=False)
        samples = np.random.choice(TS, self.batch_size, replace=False)
        #for i in sample_indx:
        for sample in samples:
            #print(sample)
            r = sample['reward']
            s = sample['Current State']
            s_old = sample['Old State']
            a = sample['action']

            if r == -10:
                y = r
            else:
                y = r + self.discount * np.max(self.Network.run(s))
            gradient = np.zeros(4)
            gradient[a] = (y - self.Network.run(s_old)[a]) ** 2
            #print(gradient, y, self.Network.run(s_old)[a])
            self.Network.backprop(gradient)
            self.Network.update(self.learning_rate)


SnakeFieldSizeX  = 30
SnakeFieldSizeY  = 30
runs             = 10000
discount         = 0.9
learning_rate    = 1
epsilon_start    = 0.9
epsilon_max      = 0.9
epsilon_increase = 0.1
memory_size      = 500
batch_size       = 400

a = QLearning(discount, learning_rate, epsilon_start, epsilon_max, epsilon_increase, memory_size, batch_size, SnakeFieldSizeX, SnakeFieldSizeY)
a.initialize_NN(6, [6,4], ['ReLU','ReLU'])
#Snake = Snake_Q.SnakeQ(60,60)
#Snake.start()
#Snake.move(a.Network)

a.run(runs)
#a.Network.printNN()
