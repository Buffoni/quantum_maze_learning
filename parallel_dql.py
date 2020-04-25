#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Function caller for multiprocessing. Runs multiple processes with Pool with arguments taken from the list parameters.
In particular, the variable parallel_instances defines how many instances for each set of the other parameter are
called.

Defines a 'test mode' with the flag debug_test, which allows to run with simplified parameters when debug_test=True.

@author: Nicola Dalla Pozza
"""
import datetime
import torch.multiprocessing as mp
import smtplib
import time

# Wrapper
from learning_tools import *


def function_wrapper(x):
    p, time_samples = x['p'], x['time_samples']
    print('deep_Q_learning_maze has started with p =', p, ', time_samples =', time_samples)
    filename, elapsed, _, _ = deep_Q_learning_maze(**x)
    print('Saved files', filename, ', elapsed time', elapsed, 'sec.\n', flush=True)
    return filename, elapsed


if __name__ == '__main__':
    tic = time.time()
    today = datetime.datetime.now()
    print(str(today))
    codeName = 'par_python'
    print(codeName, 'is running...')

    # CODE
    debug_test = False
    if debug_test:
        pVect = np.arange(0.1, 0.3, 0.1)  #
        time_samples_vect = np.array([10, 20], dtype=int)
        total_actions = 4  #
        num_episodes = 50  # 50
        batch_size = 128
        replay_capacity = batch_size
        eps_decay = 1500
        target_update = 10
        maze_filename = 'maze-zigzag-4x4-1.pkl'
        action_selector = None
        diag_threshold = 10 ** (-12)
        link_update = 0.5
        action_mode = 'reverse'  # 'reverse', 'sum', 'subtract'
        parallel_instances = 2
    else:
        pVect = np.array([0, 0.4])  # np.array([0, 0.4, 0.7, 1]) #
        time_samples_vect = np.array([350, 700, 1400, 2800], dtype=int)  # np.array([10, 20], dtype=int) #
        total_actions = 4  #
        num_episodes = 3000  # 80000  # 50
        batch_size = 128
        replay_capacity = 2000
        eps_decay = 1500
        target_update = 10
        maze_filename = 'maze-zigzag-4x4-1.pkl'
        # action_selector = None
        # diag_threshold = 10 ** (-12)
        action_selector = 'probability_mask'
        diag_threshold = 10 ** (-12)
        # action_selector = 'threshold_mask'
        # diag_threshold = 10 ** (-4)
        link_update = 0.5
        action_mode = 'sum'  # 'reverse', 'sum', 'subtract'
        parallel_instances = 5

    print('pVect =', pVect)
    print('time_samples_vect =', time_samples_vect)
    print('total_actions =', total_actions)
    print('num_episodes =', num_episodes)
    print('batch_size =', batch_size)
    print('replay_capacity =', replay_capacity)
    print('eps_decay =', eps_decay)
    print('target_update =', target_update)
    print('maze_filename =', maze_filename)
    print('action_selector =', action_selector)
    print('diag_threshold =', diag_threshold)
    print('link_update =', link_update)
    print('action_mode =', action_mode)
    print('parallel_instances =', parallel_instances)

    # parameters = [(p, time_samples, total_actions, num_episodes) for p in pVect for time_samples in time_samples_vect]
    parameters = [[{'p': p, 'time_samples': time_samples, 'total_actions': total_actions, 'num_episodes': num_episodes,
                    'replay_capacity': replay_capacity, 'batch_size': batch_size, 'eps_decay': eps_decay,
                    'target_update': target_update, 'maze_filename': maze_filename, 'action_selector': action_selector,
                    'diag_threshold': diag_threshold, 'link_update': link_update, 'action_mode': action_mode,
                    'save_filename': str(today.strftime('%Y-%m-%d_%H-%M-%S_')) + codeName + '_EP{0}_A{1}_T{2}_P{'
                                                                                            '3:02.0f}-{4:d}'.format(
                        num_episodes, total_actions, time_samples, 10 * p, instance_number)}]
                  for p in pVect for time_samples in time_samples_vect for instance_number in range(parallel_instances)]

    parameters.sort(key=lambda par: par[0]['time_samples'], reverse=True)

    parallel_processes = min(20, mp.cpu_count())
    with mp.Pool(parallel_processes) as pool:
        results = pool.starmap_async(function_wrapper, parameters).get() # ritorna una lista di tuple con i risultati

    #  Timing
    toc = time.time()
    elapsedTime = round(toc - tic)
    print('Total elapsed time:', str(elapsedTime), 'sec.')

    # Sending concluding email
    sender = 'np@unifi.it'
    receivers = ['nicola.dallapozza@unifi.it']
    message = """Subject: Simulation of {codeName} concluded. 
    
    Python simulation concluded.
    Elapsed time {elapsedTime} sec.
    
    """
    message = message.format(elapsedTime=elapsedTime, codeName=codeName)
    server = smtplib.SMTP('out.unifi.it:25')
    server.sendmail(sender, receivers, message)
    server.quit()

    print('Email sent. Job concluded.')
