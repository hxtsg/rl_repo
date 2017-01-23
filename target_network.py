import sys
import cv2
import cv2.cv as cv
import pickle
import pygame
sys.path.append("game/")
from pygame.locals import *
import gym
import tensorflow as tf
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
import wrapped_flappy_bird as game

CNN_INPUT_WIDTH = 80
CNN_INPUT_HEIGHT = 80
CNN_INPUT_DEPTH = 1
SERIES_LENGTH = 4

REWARD_COFF = 3.0

INITIAL_EPSILON = 1
FINAL_EPSILON = 0.0001
REPLAY_SIZE = 50000
BATCH_SIZE = 32
GAMMA = 0.99
OBSERVE_TIME = 500
ENV_NAME = 'Breakout-v3'
EPISODE = 10000
STEP = 5000
TEST = 10
COPY_TARGET_PERIOD = 50

class Toolkit():
    def __init__(self):
        self.fileName = "human_play_buffer"
    def WriteReplayBufferToFile(self,replay_buffer):
        pickle.dump( replay_buffer, open( self.fileName, "wb" ), True )
        print "Written Done!"
        return
    def ReadReplayBufferFromFile(self):
        buffer = pickle.load( open( self.fileName, "rb" ) )
        print "Read Done!"
        return buffer


class ImageProcess():
    def ColorMat2B(self,state):   # this is the function used for the game flappy bird
        height = 80
        width = 80
        state_gray = cv2.cvtColor( cv2.resize( state, ( height, width ) ) , cv2.COLOR_BGR2GRAY )
        _,state_binary = cv2.threshold( state_gray, 5, 255, cv2.THRESH_BINARY )
        state_binarySmall = cv2.resize( state_binary, ( width, height ))
        cnn_inputImage = state_binarySmall.reshape( ( height, width ) )
        return cnn_inputImage

    def ColorMat2Binary(self, state):
        # state_output = tf.image.rgb_to_grayscale(state_input)
        # state_output = tf.image.crop_to_bounding_box(state_output, 34, 0, 160, 160)
        # state_output = tf.image.resize_images(state_output, 80, 80, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # state_output = tf.squeeze(state_output)
        # return state_output

        height = state.shape[0]
        width = state.shape[1]
        nchannel = state.shape[2]

        sHeight = int(height * 0.5)
        sWidth = CNN_INPUT_WIDTH

        state_gray = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
        # print state_gray.shape
        # cv2.imshow('test2', state_gray)
        # cv2.waitKey(0)

        _, state_binary = cv2.threshold(state_gray, 5, 255, cv2.THRESH_BINARY)

        state_binarySmall = cv2.resize(state_binary, (sWidth, sHeight), interpolation=cv2.INTER_AREA)

        cnn_inputImg = state_binarySmall[25:, :]
        # rstArray = state_graySmall.reshape(sWidth * sHeight)
        cnn_inputImg = cnn_inputImg.reshape((CNN_INPUT_WIDTH, CNN_INPUT_HEIGHT))
        # print cnn_inputImg.shape

        return cnn_inputImg

    def ShowImageFromNdarray(self, state, p):
        imgs = np.ndarray(shape=(4, 80, 80))

        for i in range(0, 80):
            for j in range(0, 80):
                for k in range(0, 4):
                    imgs[k][i][j] = state[i][j][k]

        cv2.imshow(str(p + 1), imgs[0])
        cv2.imshow(str(p + 2), imgs[1])
        cv2.imshow(str(p + 3), imgs[2])
        cv2.imshow(str(p + 4), imgs[3])


class DQN():
    def __init__(self, env):
        self.imageProcess = ImageProcess()
        self.toolkit = Toolkit()
        self.epsilon = INITIAL_EPSILON

        self.replay_buffer = deque()

        self.human_buffer = None

        self.test_data = []  # a list for reward in each episode

        self.recent_history_queue = deque()
        self.action_dim = 2
        self.state_dim = CNN_INPUT_HEIGHT * CNN_INPUT_WIDTH
        self.time_step = 0
        self.OBSERVE_WRITTEN = False
        self.session = tf.InteractiveSession()
        self.create_network_value()

        # self.create_network_target()

        # self.create_training_method()
        self.observe_time = 0

        self.merged = tf.merge_all_summaries()
        self.summary_writer = tf.train.SummaryWriter('/path/to/logs', self.session.graph)

        self.session.run(tf.initialize_all_variables())


    def create_network_target(self):

        INPUT_DEPTH = SERIES_LENGTH

        self.input_layer_target = tf.placeholder(tf.float32, [None, CNN_INPUT_WIDTH, CNN_INPUT_HEIGHT, INPUT_DEPTH])
        self.action_input_target = tf.placeholder(tf.float32, [None, self.action_dim])
        self.y_input_target = tf.placeholder(tf.float32, [None])

        self.W21 = self.get_weights([8, 8, 4, 32])
        self.b21 = self.get_bias([32])

        h_conv21 = tf.nn.relu(tf.nn.conv2d(self.input_layer_target, self.W21, strides=[1, 4, 4, 1], padding='SAME') + self.b21)
        conv21 = tf.nn.max_pool(h_conv21, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        self.W22 = self.get_weights([4, 4, 32, 64])
        self.b22 = self.get_bias([64])

        h_conv22 = tf.nn.relu(tf.nn.conv2d(conv21,self.W22, strides=[1, 2, 2, 1], padding='SAME') + self.b22)
        # conv2 = tf.nn.max_pool( h_conv2, ksize = [ 1, 2, 2, 1 ], strides= [ 1, 2, 2, 1 ], padding= 'SAME' )

        self.W23 = self.get_weights([3, 3, 64, 64])
        self.b23 = self.get_bias([64])

        h_conv23 = tf.nn.relu(tf.nn.conv2d(h_conv22, self.W23, strides=[1, 1, 1, 1], padding='SAME') + self.b23)
        # conv3 = tf.nn.max_pool( h_conv3, ksize= [ 1,2,2,1], strides=[ 1,2,2,1 ],padding= 'SAME' )




        self.W_fc21 = self.get_weights([1600, 512])
        self.b_fc21 = self.get_bias([512])

        # h_conv2_flat = tf.reshape( h_conv2, [ -1, 11 * 11 * 32 ] )
        conv23_flat = tf.reshape(h_conv23, [-1, 1600])

        h_fc21 = tf.nn.relu(tf.matmul(conv23_flat, self.W_fc21) + self.b_fc21)

        W_fc22 = self.get_weights([512, self.action_dim])
        b_fc22 = self.get_bias([self.action_dim])

        self.Q_value_target = tf.matmul(h_fc21, W_fc22) + b_fc22
        Q_action_target = tf.reduce_sum(tf.mul(self.Q_value_target, self.action_input_target), reduction_indices=1)
        self.cost_target = tf.reduce_mean(tf.square(self.y_input_target - Q_action_target))

        self.optimizer_target = tf.train.AdamOptimizer(1e-6).minimize(self.cost_target)



    def create_network_value(self):

        INPUT_DEPTH = SERIES_LENGTH

        self.input_layer_value = tf.placeholder(tf.float32, [None, CNN_INPUT_WIDTH, CNN_INPUT_HEIGHT, INPUT_DEPTH],
                                          )
        self.action_input_value = tf.placeholder(tf.float32, [None, self.action_dim])
        self.y_input_value = tf.placeholder(tf.float32, [None])

        self.W11 = self.get_weights([8, 8, 4, 32])
        self.b11 = self.get_bias([32])

        h_conv11 = tf.nn.relu(tf.nn.conv2d(self.input_layer_value, self.W11, strides=[1, 4, 4, 1], padding='SAME') + self.b11)
        conv11 = tf.nn.max_pool(h_conv11, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        self.W12 = self.get_weights([4, 4, 32, 64])
        self.b12 = self.get_bias([64])

        h_conv12 = tf.nn.relu(tf.nn.conv2d(conv11, self.W12, strides=[1, 2, 2, 1], padding='SAME') + self.b12)
        # conv2 = tf.nn.max_pool( h_conv2, ksize = [ 1, 2, 2, 1 ], strides= [ 1, 2, 2, 1 ], padding= 'SAME' )

        self.W13 = self.get_weights([3, 3, 64, 64])
        self.b13 = self.get_bias([64])

        h_conv13 = tf.nn.relu(tf.nn.conv2d(h_conv12, self.W13, strides=[1, 1, 1, 1], padding='SAME') + self.b13)
        # conv3 = tf.nn.max_pool( h_conv3, ksize= [ 1,2,2,1], strides=[ 1,2,2,1 ],padding= 'SAME' )




        self.W_fc11 = self.get_weights([1600, 512])
        self.b_fc11 = self.get_bias([512])

        # h_conv2_flat = tf.reshape( h_conv2, [ -1, 11 * 11 * 32 ] )
        conv13_flat = tf.reshape(h_conv13, [-1, 1600])

        h_fc11 = tf.nn.relu(tf.matmul(conv13_flat, self.W_fc11) + self.b_fc11)

        self.W_fc12 = self.get_weights([512, self.action_dim])
        self.b_fc12 = self.get_bias([self.action_dim])

        self.Q_value_value = tf.matmul(h_fc11, self.W_fc12) + self.b_fc12
        Q_action_value = tf.reduce_sum(tf.mul(self.Q_value_value, self.action_input_value), reduction_indices=1)
        self.cost_value = tf.reduce_mean(tf.square(self.y_input_value - Q_action_value))

        self.optimizer_value = tf.train.AdamOptimizer(1e-6).minimize(self.cost_value)

    # def create_training_method(self):
    #
    # 	# if len(self.recent_history_queue) > 4:
    # 	# 	sess = tf.Session()
    # 	# 	print sess.run(self.Q_value)
    # 	# global_step = tf.Variable(0, name='global_step', trainable=True)
    # 	# self.optimizer = tf.train.AdamOptimizer( 0.001 ).minimize( self.cost )

    def copy_params(self):   # copy the params from value network to target_network
        self.W21 = tf.Variable( self.W11 )
        self.b21 = tf.Variable( self.b11 )

        self.W22 = tf.Variable( self.W12 )
        self.b22 = tf.Variable( self.b12 )

        self.W23 = tf.Variable( self.W13 )
        self.b23 = tf.Variable( self.b13 )

        self.W_fc21 = tf.Variable( self.W_fc11)
        self.b_fc21 = tf.Variable( self.b_fc11 )

        self.W_fc22 = tf.Variable( self.W_fc12 )
        self.b_fc22 = tf.Variable( self.b_fc12 )
        print 'Copy Done!'

    def train_network(self):
        self.time_step += 1

        minibatch = random.sample(self.replay_buffer, BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]
        done_batch = [data[4] for data in minibatch]
        # self.imageProcess.ShowImageFromNdarray( state_batch[0], 1 )




        y_batch = []
        Q_value_batch = self.Q_value_value.eval(feed_dict={self.input_layer_value: next_state_batch})

        for i in xrange(BATCH_SIZE):
            if done_batch[i]:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + GAMMA * np.max(Q_value_batch[i]))

        self.optimizer_value.run(feed_dict={

            self.input_layer_value: state_batch,
            self.action_input_value: action_batch,
            self.y_input_value: y_batch

        })


    def InitFromHumanBehavior(self):   # Init the network from reading the human behavior recorded in the file
        self.replay_buffer = self.toolkit.ReadReplayBufferFromFile()
        TRAINTIME = 100
        for i in range(TRAINTIME):
            self.train_network()

    def percieve(self, state_shadow, action, reward, state_shadow_next, done, time_step):

        #
        # state_of_shadow = self.getRecentHistory_stack(state, append_or_not=True)
        # state_of_shadow_next = self.getRecentHistory_stack(next_state, append_or_not=False)


        self.replay_buffer.append([state_shadow, action, reward, state_shadow_next, done])

        self.observe_time += 1


        if len(self.replay_buffer) > REPLAY_SIZE:
            self.replay_buffer.popleft()

        if len(self.replay_buffer) > BATCH_SIZE and self.observe_time > OBSERVE_TIME:
            self.train_network()

    def get_greedy_action(self, state_shadow):

        rst = self.Q_value_value.eval(feed_dict={self.input_layer_value: [state_shadow]})[0]
        # print rst

        return np.argmax(rst)

    def get_action(self, state_shadow):
        if self.epsilon >= FINAL_EPSILON and self.observe_time > OBSERVE_TIME:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000

        action = np.zeros(self.action_dim)
        action_index = None
        if random.random() < self.epsilon:
            action_index = random.randint(0, self.action_dim - 1)
        else:
            action_index = self.get_greedy_action(state_shadow)

        action[ action_index ] = 1
        return action


    def get_human_action(self):
        action = np.zeros( self.action_dim )

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
        keys = pygame.key.get_pressed()
        if keys[K_ESCAPE]:
            sys.exit()

        if keys[K_SPACE]:
            action[ 1 ] = 1
        else:
            action[ 0 ] = 1
        return action

    def get_weights(self, shape):
        weight = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(weight)

    def get_bias(self, shape):
        bias = tf.constant(0.01, shape=shape)
        return tf.Variable(bias)


def main():
    env = gym.make(ENV_NAME)
    state_shadow = None
    next_state_shadow = None
    pygame.init()

    plt.axis([0, 10000, 0, 200])

    plt.ion()

    agent = DQN(env)
    total_reward_decade = 0

    game_state = game.GameState()
    action = np.zeros( agent.action_dim )
    action[ 1 ] = 1

    state, reward, done = game_state.frame_step(action)

    state = agent.imageProcess.ColorMat2B( state )  # now state is a binary image of 80 * 80

    state_shadow = np.stack( ( state, state, state, state ), axis = 2 )



    for episode in xrange(EPISODE):
        total_reward = 0
        # if episode % COPY_TARGET_PERIOD == 0:
            # agent.copy_params()
        for step in xrange(STEP):

            action = agent.get_action( state_shadow )

            next_state, reward, done = game_state.frame_step(action)

            next_state = np.reshape( agent.imageProcess.ColorMat2B( next_state ), ( 80,80,1 ) )


            next_state_shadow = np.append( next_state, state_shadow[ :,:,:3 ], axis= 2 )

            total_reward += reward
            agent.percieve(state_shadow, action, reward, next_state_shadow, done, episode)
            state_shadow = next_state_shadow

            if done:
                break
        print 'Episode:', episode, 'Total Point this Episode is:', total_reward


        total_reward_decade += total_reward
        if episode % 10 == 0:
            print '-------------'
            print 'Decade:', episode / 10, 'Average Reward in this Decade is:', total_reward_decade / 10.0
            print '-------------'
            if episode != 0:
                agent.test_data.append(total_reward_decade / 10.0)
                print len( agent.test_data )
            total_reward_decade = 0



    pickle.dump( agent.test_data, open( "test_data_target_network1", "w" ), False )
    for i in range(len( agent.test_data )):
        plt.scatter( i , agent.test_data[ i ] )

    plt.waitforbuttonpress()

if __name__ == '__main__':
    main()

