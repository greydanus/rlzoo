# rlzoo
A central location for my reinforcement learning experiments

Gists
--------
For minimalist, concept-based code:
* [pong + policy gradients](https://gym.openai.com/evaluations/eval_qbonzReT72KkjMRBcziwQ)
* [cartpole + policy gradients](https://gym.openai.com/evaluations/eval_hpkg5wFHQ5WUjJnnea0wkw)

Notebooks
--------
There are several notebooks here. Some of them are under active development.

First I solved Cartpole as a toy problem (I know that pg is overkill)
![Cartpole gif](static/gym_cartpole.mov)
Then I used the same algorithm to solve something trickier - Atari Pong!
![Pong gif](static/gym_pong.gif)

Code for both of these was derived in part from karpathy's excellent [deep rl post](https://karpathy.github.io/2016/05/31/rl/).

Dependencies
--------
* Mostly Python 2.7. You will need:
 * Numpy
 * Matplotlib
 * [TensorFlow](https://www.tensorflow.org/versions/master/get_started/os_setup.html#pip_install)
 * OpenAI [Gym](https://gym.openai.com/)
