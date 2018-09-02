import numpy as np
import NN
import Snake_Q
import time
import pygame


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

    def __init__(self, discount, learning_rate, epsilon_start, epsilon_max, epsilon_increase, memory_size, batch_size, SnakeFieldSizeX , SnakeFieldSizeY):
        self.discount         = discount
        self.learning_rate    = learning_rate                                   # alpha
        self.epsilon_start    = epsilon_start
        self.epsilon_max      = epsilon_max
        self.epsilon_increase = epsilon_increase
        self.memory_size      = memory_size
        self.batch_size       = batch_size

        self.SnakeFieldSizeX  = SnakeFieldSizeX
        self.SnakeFieldSizeY  = SnakeFieldSizeY

    def initialize_NN(self, n_inputs, neuronCounts, activation_functions):
        self.Network = NN.NN(n_inputs, neuronCounts, activation_functions)





    def train(self, runs):
        Training_set = {"Old State" : [], "Current State": [], "reward": []}
        run_time = 0
        TS_index = 0
        Snake = Snake_Q.SnakeQ(self.SnakeFieldSizeX,self.SnakeFieldSizeY)

        while run_time < runs:
            score = 0
            Score = []
            Snake.start()
            pygame.init()
            screen = pygame.display.set_mode(Snake.size)

            clock = pygame.time.Clock() 
            for row in range(Snake.FieldSizeX):
                    for column in range(Snake.FieldSizeY):
                        color = get_RGB(Snake.grid, row, column)
                        pygame.draw.rect(screen, color, [(Snake.rec.l + Snake.rec.w) * column + Snake.rec.l + 10,
                                                         (Snake.rec.l + Snake.rec.h) * row + Snake.rec.l + 10, Snake.rec.w, Snake.rec.h])


            pygame.display.flip()
            clock.tick(Snake.FPS)
            while Snake.alive:
                # Get Input for the NN
                Input = get_input(Snake,self.SnakeFieldSizeX, self.SnakeFieldSizeY )
                Training_set["Old State"].append(Input)
                # Run NN
                #print('Input',Input)
                Output = self.Network.run_NN(Input)
                #print('Output',Output)
                # Get direction (highest output value, 1 neuron left, second neuron up,
                # third neuron right, fourth neuron down)
                key = np.where(Output== np.max(max(Output)))[0][0]

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
                if TS_index == memory_size:
                    TS_index = 0

                if TS_index == batch_size:
                    train

                TS_index += 1
                clock.tick(Snake.FPS)
                pygame.display.flip()
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_:
                            return 1

                for row in range(Snake.FieldSizeX):
                    for column in range(Snake.FieldSizeY):
                        color = get_RGB(Snake.grid, row, column)
                        pygame.draw.rect(screen, color, [(Snake.rec.l + Snake.rec.w) * column + Snake.rec.l + 10,
                                                         (Snake.rec.l + Snake.rec.h) * row + Snake.rec.l + 10, Snake.rec.w, Snake.rec.h])

            run_time += 1


            Score.append(Snake.score)
            print(np.mean(Score))
            print(Snake.score)


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

a.train(runs)
#a.Network.printNN()