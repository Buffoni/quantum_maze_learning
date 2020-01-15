from gym.envs.registration import register

register(
    id='quantum-maze-v0',
    entry_point='gym_quantum_maze.envs:QuantumMazeEnv',
)