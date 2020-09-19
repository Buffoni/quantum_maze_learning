import os
import fnmatch
import pickle

inputDir = "data"
outputDir = "dataNoEnv"

if not os.path.isdir(outputDir):
    os.mkdir(outputDir)

for file in [file for file in os.listdir(inputDir) if fnmatch.fnmatch(file, "*.pkl") and file != "maze_8x8.pkl"]:
    with open(os.path.join(inputDir, file), 'rb') as f:
        [episode_transfer_to_sink, env, steps_done, maze_filename, p, time_samples, total_actions,
         num_episodes, changeable_links, batch_size, gamma, eps_start, eps_end, eps_decay, target_update,
         replay_capacity, reward_no_actions, reward_final, optimal_sequence, target_transfer_to_sink] = pickle.load(f)

    with open(os.path.join(outputDir, file), 'wb') as f:
        pickle.dump([episode_transfer_to_sink, steps_done, maze_filename, p, time_samples, total_actions,
         num_episodes, changeable_links, batch_size, gamma, eps_start, eps_end, eps_decay, target_update,
         replay_capacity, reward_no_actions, reward_final, optimal_sequence, target_transfer_to_sink], f)