#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 09:00:09 2020

@author: langdon

Functions for generating trial data.
"""

import numpy as np
from scipy.sparse import random
import torch

from scipy import stats
import scipy.ndimage

import random


def generate_input_target_stream(cxt, color_coh, correct_choice, n_t,
                                 stim_on,stim_off, dec_on, dec_off, noise_factor):
    """
    Generate input and target sequence for a given set of trial conditions.

    :param t:
    :param tau:
    :param color_coh:
    :param baseline:
    :param alpha:
    :param sigma_in:
    :param stim_on:
    :param stim_off:
    :param dec_off:
    :param dec_on:

    :return: input stream
    :return: target stream

    """

    # input: 4 dimensions
    # 0: left target(1:red; 0: green); 1: right target(1:red; 0: green); 2: color coherence for red; 3: color cohernece for green

    # # Convert trial events to discrete time

    # # Transform coherence to signal
    # motion_r = (1 + color_coh) / 2
    # motion_l = 1 - motion_r


    # # Motion input stream: 0: right motion coh; 1: left motion coh
    # motion_input = np.zeros([n_t, 2])
    # motion_input[stim_on - 1:stim_off, 0] = motion_r * np.ones([stim_off - stim_on + 1])
    # motion_input[stim_on - 1:stim_off, 1] = motion_l * np.ones([stim_off - stim_on + 1])

    # # Input stream is rectified sum of baseline, task and noise signals.
    # input_stream = np.maximum( motion_input, 0)



    input_stream = np.zeros([n_t, 4])
    if cxt == 0:
        input_stream[stim_on:stim_off,0] = 1
        input_stream[stim_on:stim_off,1] = 0.1
    else:
        input_stream[stim_on:stim_off,0] = 0.1
        input_stream[stim_on:stim_off,1] = 1       


    if color_coh > 0:
        input_stream[dec_on:dec_off,2] = color_coh
        input_stream[dec_on:dec_off,3] = 0

    else:
        input_stream[dec_on:dec_off,3] = -color_coh
        input_stream[dec_on:dec_off,2] = 0


    # add a random noise
    coh_noise1 = np.random.normal(0, noise_factor, size=n_t)
    coh_noise1[:dec_on] = 0
    input_stream[:,2] = input_stream[:,2] + coh_noise1
    coh_noise2 = np.random.normal(0, noise_factor, size=n_t)
    coh_noise2[:dec_on] = 0
    input_stream[:,3] = input_stream[:,3] + coh_noise2



    # Target stream: 0: left; 1: right
    target_stream = 0.2 * np.ones([n_t, 2])
    if correct_choice == -1:
        target_stream[dec_on - 1:dec_off, 0] = 1.2 * np.ones([dec_off - dec_on + 1, 1]).squeeze()
        target_stream[dec_on - 1:dec_off, 1] = 0.2 * np.ones([dec_off - dec_on + 1, 1]).squeeze()
    else:
        target_stream[dec_on - 1:dec_off, 0] = 0.2 * np.ones([dec_off - dec_on + 1, 1]).squeeze()
        target_stream[dec_on - 1:dec_off, 1] = 1.2 * np.ones([dec_off - dec_on + 1, 1]).squeeze()

    return input_stream, target_stream


def generate_trials( n_trials,stim_on = 10, dec_on = 80, n_t = 130, noise_factor = 0.2):
    """
    Create a set of trials consisting of inputs, targets and trial conditions.

    :param tau:
    :param trial_events:
    :param n_trials: number of trials per condition.
    :param alpha:
    :param sigma_in:
    :param baseline:
    :param n_coh:

    :return: dataset
    :return: mask
    :return: conditions: array of dict
    """

    #cohs = np.hstack((-10 ** np.linspace(0, -2, n_coh), 10 ** np.linspace(-2, 0, n_coh)))
    

     # coh: <0: green dominate; >0: red dominant
     # cxt: 0: RL&GR; 1: RR&GL
     # correct_choice: -1: left; 1:right

    cohs = [-1.  -0.75, -0.5  , -0.25  , -0.125 , -0.0625,  0.0625,  0.125 ,
        0.25  ,  0.5, 0.75, 1.]

    # stim_on= int(round(n_t * .4))
    stim_off= int(round(n_t))
    # dec_on= int(round(n_t * .75))
    dec_off= int(round(n_t))


    inputs = []
    targets = []
    conditions = []
    for color_coh in cohs:

        for i in range(n_trials):
            cxt = random.choice([0,1])
            correct_choice = -1 if ((color_coh > 0 and cxt == 0) or (color_coh < 0 and cxt == 1)) else 1




            conditions.append({'color_coh': color_coh, 'cxt': cxt,  'correct_choice':correct_choice})
            input_stream, target_stream = generate_input_target_stream(cxt, color_coh,correct_choice,
                                                                       n_t,
                                                                       stim_on,
                                                                       stim_off,
                                                                       dec_on,
                                                                       dec_off, noise_factor)
            inputs.append(input_stream)
            targets.append(target_stream)
    inputs = np.stack(inputs, 0)
    targets = np.stack(targets, 0)

    perm = np.random.permutation(len(inputs))
    inputs = torch.tensor(inputs[perm, :, :]).float()
    targets = torch.tensor(targets[perm, :, :]).float()
    conditions = [conditions[index] for index in perm]


    mask = torch.ones_like(targets)
    mask[:,:dec_on,:] = 0

    return inputs, targets, mask, conditions




















