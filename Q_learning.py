import numpy as np
import NN
import Snake_Q
import time
import pygame
import random
import psutil
import os

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
    #print(Snake.snake)
    #print(np.shape(Snake.snake)[0])
    for index in range(np.shape(Snake.snake)[0]):
        #print("Index:", index)
        part = Snake.snake[index]
        #print(index)
        if index == 0:
            MinLeft  = xHead
            MinUp    = yHead
            MinRight = SnakeFieldSizeX-(xHead+1)
            MinDown  = SnakeFieldSizeY-(yHead+1)
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
    Input[4] = Snake.snake[0][0]-Snake.food[0]
    Input[5] = Snake.snake[0][1]-Snake.food[1]
    return Input


def get_input_images(Snake):
    Input = np.zeros((Snake.FieldSizeX, Snake.FieldSizeY, 3))   # original field doesn't include walls

    for i in range(Snake.FieldSizeX):
        for j in range(Snake.FieldSizeY):
            value = Snake.grid[i][j]
            if value == 1:
                Input[i, j, 0] = 1
            elif value == 2:
                Input[i, j, 1] = 1
            elif value == 3:
                Input[i, j, 2] = 1

    Input_padded = np.empty((Snake.FieldSizeX+2, Snake.FieldSizeY+2, 3))
    #print(Input_padded.shape)
    Input_padded[:,:,0] = np.pad(Input[:,:,0], 1, 'constant', constant_values=(1, 1))
    Input_padded[:,:,1] = np.pad(Input[:,:,1], 1, 'constant', constant_values=(0, 0))
    Input_padded[:,:,2] = np.pad(Input[:,:,2], 1, 'constant', constant_values=(0, 0))

    return Input_padded




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
        self.learning         = True

        self.SnakeFieldSizeX  = SnakeFieldSizeX
        self.SnakeFieldSizeY  = SnakeFieldSizeY
        self.timeout = SnakeFieldSizeX * SnakeFieldSizeY    # Number of moves after which snake dies without food

    def initialize_NN(self, n_inputs, neuronCounts, activation_functions):
        self.Network = NN.NN(n_inputs, neuronCounts, activation_functions)


    def run(self, runs):
        Training_set = []
        run_time = 0
        memory_index = 0
        batch_index = 0
        Snake = Snake_Q.SnakeQ(self.SnakeFieldSizeX,self.SnakeFieldSizeY)
        Score = []
        last_key = -0.5

        while run_time < runs:
            Snake.reward = 0
            timeout_counter = 0
            lifetime = 0
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
                        elif event.key == pygame.K_LEFT:
                            self.epsilon -= self.epsilon_increase
                        elif event.key == pygame.K_RIGHT:
                            self.epsilon += self.epsilon_increase
                        elif event.key == pygame.K_SPACE:
                            if self.learning:
                                epsilon_temp = self.epsilon
                                self.epsilon = 1
                            else:
                                self.epsilon = epsilon_temp
                            self.learning = not self.learning

                if len(Training_set) <= memory_index:
                    Training_set.append({})

                # Get Input for the NN
                #Input = get_input(Snake,self.SnakeFieldSizeX, self.SnakeFieldSizeY )
                Input = get_input_images(Snake)
                Training_set[memory_index]["Old State"] = Input
                #print('Input',Input)

                if random.random() < self.epsilon:
                    # Run NN
                    Output = self.Network.run(Input)
                    # Get direction (highest output value, first neuron left, second neuron up,
                    # third neuron right, fourth neuron down)
                    key = np.argmax(Output)
                    print('Output:', Output, "   key:", key)
                else:
                    options = [i for i in range(4) if not i == (last_key+2)%4]
                    key = np.random.choice(options)


                # Move snake in the direction
                Snake.move(key)
                if Snake.alive and Snake.score_old < Snake.score:
                    # Reward is 1 if our score increased
                    r = 200
                elif Snake.alive:
                    # nothing happens score is decreased for punishing long ways
                    r = 0.1
                elif Snake.alive == False:
                    # Score decreased if the snake died
                    r = -100
                Snake.reward += r
                #Input = get_input(Snake, self.SnakeFieldSizeX, self.SnakeFieldSizeY)
                Input = get_input_images(Snake)
                Training_set[memory_index]["Current State"] = Input
                Training_set[memory_index]["reward"] = r
                Training_set[memory_index]["action"] = key

                if self.learning:
                    memory_index += 1
                    batch_index  += 1
                    last_key = key

                    if memory_index == self.memory_size:
                        memory_index = 0
                    if batch_index == self.batch_size:
                        batch_index = 0
                        self.train(Training_set)

                lifetime += 1
                if Snake.score_old < Snake.score:
                    timeout_counter = 0
                else:
                    timeout_counter += 1
                if timeout_counter > self.timeout:
                    break


            run_time += 1

            self.Network.save_NN('test')

            Score.append(Snake.score)

            print("Run: {} | Score: {:>2} | Lifetime: {:>4} | Epsilon: {:.2f} | Learning: {}".format(
                    run_time, Snake.score, lifetime, self.epsilon, self.learning))
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

            if r == -100:
                y = r
            else:
                y = r + self.discount * np.max(self.Network.run(s))
            gradient = np.zeros(4)
            gradient[a] = -(y - self.Network.run(s_old)[a])
            #print(gradient, y, self.Network.run(s_old)[a])
            self.Network.backprop(gradient)
            self.Network.update(self.learning_rate / self.batch_size)
        #self.Network.printNN()


if __name__ == "__main__":

    p = psutil.Process(os.getpid())
    if(os.name == 'nt'):
        #Windows
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    elif(os.name == "posix"):
        #Unix
        p.nice(19)

    SnakeFieldSizeX  = 10
    SnakeFieldSizeY  = 10
    runs             = 1000000
    discount         = 0.9
    learning_rate    = 0.0001
    epsilon_start    = 0.05
    epsilon_max      = 0.9
    epsilon_increase = 0.01
    memory_size      = 50
    batch_size       = 40

    a = QLearning(discount, learning_rate, epsilon_start, epsilon_max, epsilon_increase, memory_size, batch_size, SnakeFieldSizeX, SnakeFieldSizeY)
    #a.initialize_NN(6, [32, 16 ,4], ['ReLU','ReLU','identity'])
    #a.initialize_NN((15,15), [(8,8,3,3,2), (4,4,3,2,1), 16, 8, 4], ['ReLU', 'ReLU', 'ReLU', 'ReLU', 'identity'])
    a.Network = NN.load_NN("test")
    #Snake = Snake_Q.SnakeQ(15,15)
    #Snake.start()
    #Snake.move(a.Network)

    a.run(runs)
    #a.Network.printNN()
