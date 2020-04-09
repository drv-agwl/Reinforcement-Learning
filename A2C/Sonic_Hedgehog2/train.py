from keras import backend as K
from keras.layers import Activation, Dense, Input, Conv2D, Lambda, Flatten
from keras.models import Model
from keras.optimizers import Adam
import numpy as np

class Agent(object):
    def __init__(self, alpha, beta, gamma = 0.99, n_actions = 7,
                 layer1_size = 1024, layer2_size = 512, input_dims = (96, 96, 4)):
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.input_dims = input_dims
        self.fc1_dims = layer1_size
        self.fc2_dims = layer2_size
        self.n_actions = n_actions
        
        self.actor, self.critic, self.policy = self.build_actor_critic_network()
        self.action_space = [i for i in range(self.n_actions)]
        
    def build_actor_critic_network(self):
        input_ = Input(shape = self.input_dims)
        input_n = Lambda(lambda x: x / 255.)(input_)

        
        conv1 = Conv2D(filters = 32, kernel_size = (8, 8), strides = (4, 4))(input_n)
        conv2 = Conv2D(filters = 64, kernel_size = (4, 4), strides = (2, 2))(conv1)
        conv3 = Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1))(conv2)
        
        flatten1 = Flatten()(conv3)
        
        delta = Input(shape = [1])
        dense1 = Dense(self.fc1_dims, activation = 'relu')(flatten1)
        dense2 = Dense(self.fc2_dims, activation = 'relu')(dense1)
        
        probs = Dense(self.n_actions, activation = 'softmax')(dense2)
        values = Dense(1, activation = 'linear')(dense2)
        
        def custom_loss(y_true, y_pred):
            out = K.clip(y_pred, 1e-8, 1-1e-8)
            log_lik = y_true*K.log(out)
            
            return K.sum(-log_lik*delta)
        
        actor = Model(input = [input_, delta], outputs =[probs])
        actor.compile(optimizer = Adam(lr = self.alpha), loss = custom_loss)
        
        critic = Model(input = [input_], output = [values])
        critic.compile(optimizer = Adam(lr = self.beta), loss = 'mean_squared_error')
        
        policy = Model(input=[input_], output = [probs])
        
        return actor, critic, policy
    
    def choose_action(self, observation):
        state = observation[np.newaxis, :]
        probabilities = self.policy.predict(state)[0]
        action = np.random.choice(self.action_space, p = probabilities)
        
        return action
    
    def learn(self, state, action, reward, state_, done):
        state = state[np.newaxis, :]
        state_ = state_[np.newaxis, :]
        
        critic_value_ = self.critic.predict(state_)
        critic_value = self.critic.predict(state)
        
        target = reward + self.gamma*critic_value_*(1 - int(done))
        delta = target - critic_value
        
        actions = np.zeros([1, self.n_actions])
        actions[np.arange(1), action] = 1.0
        
        self.actor.fit([state, delta], actions, verbose = 0)
        self.critic.fit(state, target, verbose = 0)