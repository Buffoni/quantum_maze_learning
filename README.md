# quantum_maze_learning
Reinforcement learning on a quantum maze

### Reference

Reference for the gym environment creation

<https://medium.com/@apoddar573/making-your-own-custom-environment-in-gym-c3b65ff8cdaa> 

and 

<https://github.com/MattChanTK/gym-maze/tree/master/gym_maze>

In particular, after creating a new environment, you have to register it from its folder with 

```
pip install -e .
```

### Installation test

To test the correct installation of the environment run

```
import gym
from gym_quantum_maze.envs import quantum_maze_env
env = gym.make('quantum-maze-v0')
env.reset()
for _ in range(3):
    env.render()
    action = env.action_space.sample()
    print('action=', action)
    env.step(action) # take a random action
env.close()
```



