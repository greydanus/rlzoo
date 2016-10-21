'''Solution to the Cartpole problem using Policy Gradients in Tensorflow.'''
# written October 2016 by Sam Greydanus
# inspired by gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5
import numpy as np
import gym
import tensorflow as tf

# hyperparameters
n_obs = 4              # dimensionality of observations
h = 128                # hidden layer neurons
n_actions = 2          # number of available actions
learning_rate = 1e-2   # how rapidly to update parameters
gamma = .9             # reward discount factor
decay = 0.9            # decay rate for RMSProp gradients

# gamespace
env = gym.make("CartPole-v0")
observation = env.reset()
xs,rs,ys = [],[],[]    # environment info
running_reward = 10    # worst case is ~10 for cartpole
reward_sum = 0
episode_number = 0
max_steps = 1000      # should converge around 300

# initialize model
tf_model = {}
with tf.variable_scope('layer_one',reuse=False):
    xavier_l1 = tf.truncated_normal_initializer(mean=0, stddev=1./np.sqrt(n_obs), dtype=tf.float32)
    tf_model['W1'] = tf.get_variable("W1", [n_obs, h], initializer=xavier_l1)
with tf.variable_scope('layer_two',reuse=False):
    xavier_l2 = tf.truncated_normal_initializer(mean=0, stddev=1./np.sqrt(h), dtype=tf.float32)
    tf_model['W2'] = tf.get_variable("W2", [h,n_actions], initializer=xavier_l2)

# tensorflow operations
def tf_discount_rewards(tf_r): #tf_r ~ [game_steps,1]
    discount_f = lambda a, v: a*gamma + v;
    tf_r_reverse = tf.scan(discount_f, tf.reverse(tf_r,[True, False]))
    tf_discounted_r = tf.reverse(tf_r_reverse,[True, False])
    return tf_discounted_r

def tf_policy_forward(x): #x ~ [1,D]
    h = tf.matmul(x, tf_model['W1'])
    h = tf.nn.relu(h)
    logp = tf.matmul(h, tf_model['W2'])
    p = tf.nn.softmax(logp)
    return p

# tf placeholders
tf_x = tf.placeholder(dtype=tf.float32, shape=[None, n_obs],name="tf_x")
tf_y = tf.placeholder(dtype=tf.float32, shape=[None, n_actions],name="tf_y")
tf_epr = tf.placeholder(dtype=tf.float32, shape=[None,1], name="tf_epr")

# tf reward processing (need tf_discounted_epr for policy gradient wizardry)
tf_discounted_epr = tf_discount_rewards(tf_epr)
tf_mean, tf_variance= tf.nn.moments(tf_discounted_epr, [0], shift=None, name="reward_moments")
tf_discounted_epr -= tf_mean
tf_discounted_epr /= tf.sqrt(tf_variance + 1e-6)

# tf optimizer op
tf_aprob = tf_policy_forward(tf_x)
loss = tf.nn.l2_loss(tf_y-tf_aprob) # this gradient encourages the actions taken
optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=decay)
tf_grads = optimizer.compute_gradients(loss, var_list=tf.trainable_variables(), grad_loss=tf_discounted_epr)
train_op = optimizer.apply_gradients(tf_grads)

# tf graph initialization
sess = tf.InteractiveSession()
tf.initialize_all_variables().run()

# training loop
# stop when running reward exceeds 200 (task is considered solved)
while episode_number <= max_steps and running_reward < 200:
#     if episode_number%50==0: env.render()

    # stochastically sample a policy from the network
    x = observation
    feed = {tf_x: np.reshape(x, (1,-1))}
    aprob = sess.run(tf_aprob,feed)
    aprob = aprob[0,:] # we live in a batched world :/
    
    action = np.random.choice(n_actions, p=aprob)
    label = np.zeros_like(aprob) ; label[action] = 1 # make a training 'label'

    # step the environment and get new measurements
    observation, reward, done, info = env.step(action)
    reward_sum += reward
    
    # record game history
    xs.append(x)
    ys.append(label)
    rs.append(reward)
    
    if done:
        running_reward = running_reward * 0.99 + reward_sum * 0.01
        epx = np.vstack(xs)
        epr = np.vstack(rs)
        epy = np.vstack(ys)
        xs,rs,ys = [],[],[] # reset game history
        
        feed = {tf_x: epx, tf_epr: epr, tf_y: epy}
        _ = sess.run(train_op,feed) # parameter update

        # print some updates
        if episode_number % 25 == 0:
            print 'ep: {}, reward: {}, mean reward: {:3f}'.format(
                episode_number, reward_sum, running_reward)
        
        # book-keeping
        episode_number += 1
        observation = env.reset() # reset env
        reward_sum = 0

if running_reward > 200:
    print "ep: {}: SOLVED! (running reward hit {} which is greater than 200)".format(
        episode_number, running_reward)
else:
    print "ep: {}: model did not converge. Try changing the hyperparameters.".format(episode_number)