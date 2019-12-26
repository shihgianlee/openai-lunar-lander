# Lunar Lander

![Lunar Lander](/images/openaigym.video.0.84915.video000020.gif)

This is an attempt to solve [OpenAI Lunar Lander-v2](https://gym.openai.com/envs/LunarLander-v2/) using [Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf). 

# Implementation

The search for hyperparameters values are challenging because of the large hyperparameters space need to be searched. As a result, we use the hyperparameters values from 
[Deep Q-Learning with Keras and Gym](https://keon.io/deep-q-learning/) that is used to solve [Cartpole-v1](https://gym.openai.com/envs/CartPole-v1/) as starting point. The 
[lunar_lander.py](https://github.com/shihgianlee/openai-lunar-lander/blob/master/lunar_lander.py) file has the training code for lunar lander model. For the longest time, the rewards were hovering between 0 and negative
territories. The breakthrough came when we replace epsilon-greedy exploration strategy with [Boltzman exploration](https://www.cs.cmu.edu/afs/cs/academic/class/15381-s07/www/slides/050107reinforcementLearning1.pdf) strategy.

# Result

![Lunar Lander rewards](/images/rewards.png)

# Credits

Credits are given in the [source](https://github.com/shihgianlee/openai-lunar-lander/blob/master/lunar_lander.py) and [References](#references).


# References

[1.] Deep Q-Learning with Keras and Gym. URL: https://keon.io/deep-q-learning/

[2.] Playing Atari with Deep Reinforcement Learning. URL: https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

[3.] Artificial Intelligence: Representation and Problem. URL: https://www.cs.cmu.edu/afs/cs/academic/class/15381-s07/www/slides/050107reinforcementLearning1.pdf

[4.] Human-level control through deep reinforcement learning. URL: https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf

[5.] Reinforcement Learning w/ Keras + OpenAI: DQNs. URL: https://towardsdatascience.com/reinforcement-learning-w-keras-openai-dqns-1eed3a5338c

[6.] Keras RL. URL: https://github.com/keras-rl/keras-rl
