#!/usr/bin/env python
# inference.py
# Base code by George H. Chen (georgehc@mit.edu) -- updated 10/12/2012 5:36pm
# Modified by: <your name here!>

# use this to enable/disable graphics
enable_graphics = True

import sys
import numpy as np
import robot
from robot import Distribution
if enable_graphics:
    import graphics


#-----------------------------------------------------------------------------
# Functions for you to implement
#

def helper_optimized(hidden):
    possibilities = []
    actions = ["stay","left","right","up","down"]
    for i in [-1,0,1]:
        for j in [-1,0,1]:
            for action in actions:
                if hidden[0]+i>=0 and hidden[1]+j>=0 and hidden[0]+i<=11 and hidden[1]+j<=7:
                    possibilities.append((hidden[0]+i, hidden[1]+j, action))
    return possibilities




def forward_backward(all_possible_hidden_states,
                     all_possible_observed_states,
                     prior_distribution,
                     transition_model,
                     observation_model,
                     observations):
    """
    Inputs
    ------
    all_possible_hidden_states: a list of possible hidden states

    all_possible_observed_states: a list of possible observed states

    prior_distribution: a distribution over states

    transition_model: a function that takes a hidden state and returns a
        Distribution for the next state

    observation_model: a function that takes a hidden state and returns a
        Distribution for the observation from that hidden state

    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of marginal distributions at each time step; each distribution
    should be encoded as a Distribution (see the Distribution class in
    robot.py and see how it is used in both robot.py and the function
    generate_data() above, and the i-th Distribution should correspond to time
    step i
    """

    num_time_steps = len(observations)
    forward_messages = [None] * num_time_steps
    forward_messages[0] = prior_distribution

    for timestep in range(1,num_time_steps):
        observation = observations[timestep-1]
        new_message = Distribution()
        for hidden_state_1 in all_possible_hidden_states:
            optimized_hidden = helper_optimized(hidden_state_1)

            for hidden_state_0 in optimized_hidden:
                if observation:
                    new_message[hidden_state_1] += \
                    observation_model(hidden_state_0)[observation]*forward_messages[timestep-1][hidden_state_0]\
                    *transition_model(hidden_state_0)[hidden_state_1]
                else:
                    new_message[hidden_state_1] += \
                    forward_messages[timestep-1][hidden_state_0]\
                    *transition_model(hidden_state_0)[hidden_state_1]

        new_message.renormalize()
  

        forward_messages[timestep] = new_message
    


    backward_messages = [None] * num_time_steps
    uniform_distribution = Distribution()
    hiddenSize = len(all_possible_hidden_states)
    for hidden in all_possible_hidden_states:
        uniform_distribution[hidden] = 1/hiddenSize
    backward_messages[-1] = uniform_distribution

    for timestep in range(num_time_steps,1,-1):
        observation = observations[timestep-1]
        new_message = Distribution()
        for hidden_state_0 in all_possible_hidden_states:
            optimized_hidden = helper_optimized(hidden_state_0)
            trans_0 = transition_model(hidden_state_0)
            for hidden_state_1 in optimized_hidden:
                if observation:
                    long_exp =  observation_model(hidden_state_1)[observation]
                    long_exp*=backward_messages[timestep-1][hidden_state_1]
                    long_exp*=trans_0[hidden_state_1]
                    new_message[hidden_state_0] += long_exp
                else:
                    long_exp=backward_messages[timestep-1][hidden_state_1]
                    long_exp*=trans_0[hidden_state_1]
                    new_message[hidden_state_0] += long_exp
        new_message.renormalize()
        

        backward_messages[timestep-2] = new_message



    marginals = [None] * num_time_steps # remove this
    for timestep in range(num_time_steps):
        marginal_dist = Distribution()

        for hidden_state in all_possible_hidden_states:
   
            observation = observations[timestep]
            if observation:
                marginal_dist[hidden_state] = observation_model(hidden_state)[observation]*\
                forward_messages[timestep][hidden_state]*backward_messages[timestep][hidden_state]
            else:
                marginal_dist[hidden_state] = forward_messages[timestep][hidden_state]*backward_messages[timestep][hidden_state]

        marginal_dist.renormalize()

       
        marginals[timestep] = marginal_dist
    return marginals



def Viterbi(all_possible_hidden_states,
            all_possible_observed_states,
            prior_distribution,
            transition_model,
            observation_model,
            observations):
    """
    Inputs
    ------
    See the list inputs for the function forward_backward() above.

    Output
    ------
    A list of esimated hidden states, each encoded as a tuple
    (<x>, <y>, <action>)
    """



    num_time_steps = len(observations)
    estimated_hidden_states = [None] * num_time_steps # remove this

    messages = [None] * num_time_steps
    uniform_distribution = Distribution()
    hiddenSize = len(all_possible_hidden_states)
    for hidden in all_possible_hidden_states:
        uniform_distribution[hidden] = 1
    uniform_distribution.renormalize()
    messages[0] = prior_distribution
    traceback = [dict()]*num_time_steps
    # calculate forward messages and traceback table
    for timestep in range(1,num_time_steps):
        new_message = Distribution()
        t_i = dict()
        observation = observations[timestep-1]
        for hidden_state_1 in all_possible_hidden_states:
            maxi = -1

            optimized_hidden = helper_optimized(hidden_state_1)
            for hidden_state_0 in optimized_hidden:
                if observation:
                    #observation exists
                    long_exp = observation_model(hidden_state_0)[observation]*messages[timestep-1][hidden_state_0]\
                    *transition_model(hidden_state_0)[hidden_state_1]
                    if maxi<long_exp:
                        t_i[hidden_state_1] = hidden_state_0

                        maxi = long_exp   
                else:
                    # no observation
                    long_exp = transition_model(hidden_state_0)[hidden_state_1]*messages[timestep-1][hidden_state_0]
                    if maxi<long_exp:
                        t_i[hidden_state_1] = hidden_state_0


                        maxi = long_exp 
            new_message[hidden_state_1] = maxi  
        messages[timestep] = new_message
        traceback[timestep] = t_i
        
    estimates = [None] * num_time_steps
    value  =-1
    x_n =None
    # find x_n estimate
    for hidden in all_possible_hidden_states:
        if observations[-1]:
            new_value = observation_model(hidden)[observations[-1]]*messages[-1][hidden]
        else:
            new_value = messages[-1][hidden]

        if new_value>value:
            x_n = hidden
            value = new_value
    estimates[-1] = x_n
    #find the rest of estimates
    for timestep in range(num_time_steps,1,-1):                

        estimates[timestep-2] = traceback[timestep-1][estimates[timestep-1]]

    estimated_hidden_states=estimates

    return estimated_hidden_states

def second_best(all_possible_hidden_states,
                all_possible_observed_states,
                prior_distribution,
                transition_model,
                observation_model,
                observations):
    """
    Inputs
    ------
    See the list inputs for the function forward_backward() above.

    Output
    ------
    A list of esimated hidden states, each encoded as a tuple
    (<x>, <y>, <action>)
    """



    num_time_steps = len(observations)
    estimated_hidden_states = [None] * num_time_steps # remove this

    return estimated_hidden_states


#-----------------------------------------------------------------------------
# Generating data from the hidden Markov model
#

def generate_data(initial_distribution, transition_model, observation_model,
                  num_time_steps, make_some_observations_missing=False,
                  random_seed=None):
    # generate samples from a hidden Markov model given an initial
    # distribution, transition model, observation model, and number of time
    # steps, generate samples from the corresponding hidden Markov model
    hidden_states = []
    observations  = []

    # if the random seed is not None, then this makes the randomness
    # deterministic, which may be helpful for debug purposes
    np.random.seed(random_seed)

    # draw initial state and emit an observation
    initial_state       = initial_distribution().sample()
    initial_observation = observation_model(initial_state).sample()

    hidden_states.append(initial_state)
    observations.append(initial_observation)

    for time_step in range(1, num_time_steps):
        # move the robot
        prev_state   = hidden_states[-1]
        new_state    = transition_model(prev_state).sample()

        # maybe emit an observation
        if not make_some_observations_missing:
            new_observation = observation_model(new_state).sample()
        else:
            if np.random.rand() < .1: # 0.1 prob. of observation being missing
                new_observation = None
            else:
                new_observation = observation_model(new_state).sample()

        hidden_states.append(new_state)
        observations.append(new_observation)

    return hidden_states, observations


#-----------------------------------------------------------------------------
# Main
#

if __name__ == '__main__':
    # flags
    make_some_observations_missing = False
    use_graphics                   = enable_graphics
    need_to_generate_data          = True

    # parse command line arguments
    for arg in sys.argv[1:]:
        if arg == '--missing':
            make_some_observations_missing = True
        elif arg == '--nographics':
            use_graphics = False
        elif arg.startswith('--load='):
            filename = arg[7:]
            hidden_states, observations = robot.load_data(filename)
            need_to_generate_data = False
            num_time_steps = len(hidden_states)

    # if no data is loaded, then generate new data
    if need_to_generate_data:
        num_time_steps = 100
        hidden_states, observations = \
            generate_data(robot.initial_distribution,
                          robot.transition_model,
                          robot.observation_model,
                          num_time_steps,
                          make_some_observations_missing)

    all_possible_hidden_states   = robot.get_all_hidden_states()
    all_possible_observed_states = robot.get_all_observed_states()
    prior_distribution           = robot.initial_distribution()

    print('Running forward-backward...')
    marginals = forward_backward(all_possible_hidden_states,
                                 all_possible_observed_states,
                                 prior_distribution,
                                 robot.transition_model,
                                 robot.observation_model,
                                 observations)
    print('\n')

    timestep = num_time_steps - 1
    print("Most likely parts of marginal at time %d:" % (timestep))
    if marginals[timestep] is not None:
        print(sorted(marginals[timestep].items(), key=lambda x: x[1], reverse=True)[:10])
    else:
        print('*No marginal computed*')
    print('\n')

    timestep = 0
    print("Most likely parts of marginal at time %d:" % (timestep))
    if marginals[timestep] is not None:
        print(sorted(marginals[timestep].items(), key=lambda x: x[1], reverse=True)[:10])
    else:
        print('*No marginal computed*')
    print('\n')

    print('Running Viterbi...')
    estimated_states = Viterbi(all_possible_hidden_states,
                               all_possible_observed_states,
                               prior_distribution,
                               robot.transition_model,
                               robot.observation_model,
                               observations)
    print('\n')
    print("Last 10 hidden states in the MAP estimate:")
    for time_step in range(num_time_steps - 10, num_time_steps):
        if estimated_states[time_step] is None:
            print('Missing')
        else:
            print(estimated_states[time_step])
    print('\n')

    print('Finding second-best MAP estimate...')
    estimated_states2 = second_best(all_possible_hidden_states,
                                    all_possible_observed_states,
                                    prior_distribution,
                                    robot.transition_model,
                                    robot.observation_model,
                                    observations)
    print('\n')

    print("Last 10 hidden states in the second-best MAP estimate:")
    for time_step in range(num_time_steps - 10 - 1, num_time_steps):
        if estimated_states2[time_step] is None:
            print('Missing')
        else:
            print(estimated_states2[time_step])
    print('\n')

    difference = 0
    for time_step in range(num_time_steps):
        if estimated_states[time_step] != hidden_states[time_step]:
            difference += 1
    print("Number of differences between MAP estimate and true hidden " + \
          "states:", difference)

    difference = 0
    for time_step in range(num_time_steps):
        if estimated_states2[time_step] != hidden_states[time_step]:
            difference += 1
    print("Number of differences between second-best MAP estimate and " + \
          "true hidden states:", difference)

    difference = 0
    for time_step in range(num_time_steps):
        if estimated_states[time_step] != estimated_states2[time_step]:
            difference += 1
    print("Number of differences between MAP and second-best MAP " + \
          "estimates:", difference)

    # display
    if use_graphics:
        app = graphics.playback_positions(hidden_states,
                                          observations,
                                          estimated_states,
                                          marginals)
        app.mainloop()

