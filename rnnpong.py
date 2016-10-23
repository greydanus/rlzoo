'''Solves Pong with Policy Gradients in Tensorflow.'''
# written October 2016 by Sam Greydanus
# inspired by karpathy's gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5
import numpy as np
import gym
import tensorflow as tf

class Actor():
    def __init__(self, n_obs, n_actions, hs=[10,100], batch_size = 16, tsteps = 30):
        self.n_obs = n_obs                  # dimensionality of observations
        self.h1 = hs[0]                     # number of hidden layer neurons
        self.h2 = hs[1]
        self.n_actions = n_actions          # number of available actions
        self.tsteps = tsteps
        self.batch_size = batch_size
        
        self.model = model = {}
        with tf.variable_scope('actor',reuse=False):
            xavier_l1 = tf.truncated_normal_initializer(mean=0, stddev=1./np.sqrt(n_obs), dtype=tf.float32)
            model['W1'] = tf.get_variable("W1", [n_obs, self.h1], initializer=xavier_l1)

            rnn_init = tf.truncated_normal_initializer(stddev=0.075, dtype=tf.float32)
            model['rnn'] = tf.nn.rnn_cell.LSTMCell(self.h2, state_is_tuple=True, initializer=rnn_init)
            
            model['state'] = model['rnn'].zero_state(batch_size=1, dtype=tf.float32)
            model['batch_state'] = model['rnn'].zero_state(batch_size=batch_size, dtype=tf.float32)

            xavier_l3 = tf.truncated_normal_initializer(stddev=1./np.sqrt(self.h2), dtype=tf.float32)
            model['W3'] = tf.get_variable("W3", [self.h2, n_actions], initializer=xavier_l3)

    def policy_forward(self, x, train_mode=True, reuse=False):
        tsteps = 1 if not train_mode else self.tsteps
        state = self.model['state'] if not train_mode else self.model['batch_state']

        h = tf.matmul(x, self.model['W1']) # tsteps list of [batches, H]
        h = tf.nn.relu(h)             # ReLU nonlinearity

        hs = [tf.squeeze(h_, [1]) for h_ in tf.split(1, tsteps, tf.reshape(h, [-1, tsteps, self.h1]))]
        with tf.variable_scope('agent',reuse=reuse):
            rnn_outs, state = tf.nn.seq2seq.rnn_decoder(hs, state, self.model['rnn'])
        rnn_out = tf.reshape(tf.concat(1, rnn_outs), [-1, self.h2]) #out_rnn ~ [batches*tsteps, H]
        logps = tf.matmul(rnn_out, self.model['W3'])
        p = tf.reshape(logps, [-1, n_actions]) #p ~ [batches*tsteps, 3]
        return tf.nn.softmax(p)

class Agent():
    def __init__(self, n_obs, n_actions):
        self.gamma = .99            # discount factor for reward
        self.global_step = 0
        self.xs, self.rs, self.ys = [],[],[]
        
        self.actor_lr = 3e-4        # learning rate for policy
        self.decay = 0.992
        self.n_obs = n_obs                     # dimensionality of observations
        self.n_actions = n_actions             # number of available actions
        self.save_path ='rnn_models/pong.ckpt'
        
        # make actor part of brain
        self.tsteps = 30
        self.batch_size = 16
        self.actor = Actor(n_obs=self.n_obs, hs=[10,100], n_actions=self.n_actions, \
                           batch_size=self.batch_size, tsteps=self.tsteps)
        
        #placeholders
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, n_obs],name="x")
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, n_actions],name="y")
        self.r = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="r")
        
        #gradient processing (PG magic)
        self.discounted_r = self.discount_rewards(self.r, self.gamma)
        mean, variance= tf.nn.moments(self.discounted_r, [0], shift=None, name="reward_moments")
        self.discounted_r -= mean
        self.discounted_r /= tf.sqrt(variance + 1e-6)
        
        # initialize tf graph
        self.aprob = self.actor.policy_forward(self.x, train_mode=False, reuse=False)
        self.train_aprob = self.actor.policy_forward(self.x, train_mode=True, reuse=True)
        
        self.loss = tf.nn.l2_loss(self.y-self.train_aprob)
        self.optimizer = tf.train.RMSPropOptimizer(self.actor_lr, decay=self.decay)
        self.grads = self.optimizer.compute_gradients(self.loss, \
                                    var_list=tf.trainable_variables(), grad_loss=self.discounted_r)
        self.train_op = self.optimizer.apply_gradients(self.grads)

        self.sess = tf.InteractiveSession()
        tf.initialize_all_variables().run()
        self.saver = tf.train.Saver(tf.all_variables())
    
    def act(self, x):
        feed = {self.x: x}
        aprob = self.sess.run(self.aprob, feed) ; aprob = aprob[0,:]
        action = np.random.choice(self.n_actions, p=aprob)
        
        label = np.zeros_like(aprob) ; label[action] = 1
        self.xs.append(x)
        self.ys.append(label)
        
        return action
    
    def learn(self):
        epx = np.vstack(self.xs)
        epr = np.vstack(self.rs)
        epy = np.vstack(self.ys)
        self.xs, self.rs, self.ys = [],[],[] # reset game history
        
        unit_len = self.batch_size*self.tsteps
        buffer_len = ((unit_len - epx.shape[0]%unit_len)%unit_len)
        epx_buffer = np.zeros((buffer_len,epx.shape[1]))
        epr_buffer = np.zeros((buffer_len,epr.shape[1]))
        epy_buffer = np.zeros((buffer_len,epy.shape[1]))
        
        epx = np.concatenate((epx, epx_buffer),axis=0)
        epr = np.concatenate((epr, epr_buffer),axis=0)
        epy = np.concatenate((epy, epy_buffer),axis=0)
        
        num_batches = epx.shape[0]/unit_len
        for b in range(num_batches):
            start = b*unit_len ; stop = (b+1)*unit_len
            feed = {self.x: epx[start:stop,:], self.r: epr[start:stop,:], self.y: epy[start:stop,:]}
            _ = self.sess.run(self.train_op,feed) # parameter update
        self.global_step += 1
        
    def try_load_model(self):
        load_was_success = True # yes, I'm being optimistic
        try:
            save_dir = '/'.join(self.save_path.split('/')[:-1])
            ckpt = tf.train.get_checkpoint_state(save_dir)
            load_path = ckpt.model_checkpoint_path
            self.saver.restore(self.sess, load_path)
        except:
            print "no saved model to load. starting new session"
            load_was_success = False
        else:
            print "loaded model: {}".format(load_path)
            self.saver = tf.train.Saver(tf.all_variables())
            self.global_step = int(load_path.split('-')[-1])
            
    def save(self):
        self.saver.save(self.sess, self.save_path, global_step=self.global_step)
        
    @staticmethod
    def discount_rewards(r, gamma):
        discount_f = lambda a, v: (a*gamma + v)*(1-tf.abs(v)) + (v)*tf.abs(v);
        r_reverse = tf.scan(discount_f, tf.reverse(r,[True, False]))
        discounted_r = tf.reverse(r_reverse,[True, False])
        return discounted_r

def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195] # crop
    I = I[::2,::2,0] # downsample by factor of 2
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()

n_obs = 80*80   # dimensionality of observations
n_actions = 3
agent = Agent(n_obs, n_actions)
agent.try_load_model()

env = gym.make("Pong-v0")
observation = env.reset()
prev_x = None
running_reward = -20.48 # usually starts around 10 for cartpole
reward_sum = 0
episode_number = agent.global_step


total_parameters = 0 ; print "Model overview:"
for variable in tf.trainable_variables():
    shape = variable.get_shape()
    variable_parameters = 1
    for dim in shape:
        variable_parameters *= dim.value
    print '\tvariable "{}" has {} parameters' \
        .format(variable.name, variable_parameters)
    total_parameters += variable_parameters
print "Total of {} parameters".format(total_parameters)


print 'episode {}: starting up...'.format(episode_number)
while True:
    # if True: env.render()

    # preprocess the observation
    cur_x = prepro(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros(n_obs)
    prev_x = cur_x

    # stochastically sample a policy from the network
    action = agent.act(np.reshape(x, (1,-1)))

    # step the environment and get new measurements
    observation, reward, done, info = env.step(action + 1)
    agent.rs.append(reward)
    reward_sum += reward
    
    if done:
        running_reward = running_reward * 0.99 + reward_sum * 0.01
        agent.learn()

        if episode_number % 10 == 0:
            print 'ep: {}, reward: {}, mean reward: {:3f}'.format(episode_number, reward_sum, running_reward)
        else:
            print '\tep: {}, reward: {}'.format(episode_number, reward_sum)
            
        if episode_number % 50 == 0: agent.save() ; print "SAVED MODEL #{}".format(agent.global_step)
        
        # lame stuff
        cur_x = None
        episode_number += 1 # the Next Episode
        observation = env.reset() # reset env
        reward_sum = 0