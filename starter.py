import sys

import matplotlib.pylab as plb
import numpy as np
import mountaincar
import random as rnd
import neuronalnetwork as nn
from neuronalnetwork import State
from collections import defaultdict
import argparse

def visualize_trial(agent, n_steps=200, time_step=1.0):
    """Do a trial without learning, with display.

    Parameters
    ----------
    n_steps -- number of steps to simulate for
    """

    # prepare for the visualization
    plb.ion()
    mv = mountaincar.MountainCarViewer(agent.mountain_car)
    mv.create_figure(n_steps=n_steps, max_time=n_steps)
    plb.draw()
    plb.pause(0.0001)

    # make sure the mountain-car is reset
    agent.mountain_car.reset()

    for n in range(0, n_steps):

        # choose a action
        state = State(agent.mountain_car.x, agent.mountain_car.x_d)
        action = agent.choose_action(state)
        agent.mountain_car.apply_force(action)

        print("action", action)
        # simulate the timestep
        agent.mountain_car.simulate_timesteps(100, 0.01)

        # update the visualization
        mv.update_figure()
        plb.draw()
        plb.pause(0.0001)
        # check for rewards
        if agent.mountain_car.R > 0.0:
            print("reward obtained at t = ", agent.mountain_car.t)
            break


class OptimalAgent():
    """The optimal agent for the mountain-car task.
        (No learning involved)
    """

    def __init__(self, mountain_car=None):

        if mountain_car is None:
            self.mountain_car = mountaincar.MountainCar()
        else:
            self.mountain_car = mountain_car

    def take_action(self, state):
        if state.v > 0:
            return 1
        else:
            return -1

class SarsaAgent():
    def __init__(self,warm_start=None, mountain_car=None):

        if mountain_car is None:
            self.mountain_car = mountaincar.MountainCar()
        else:
            self.mountain_car = mountain_car

        self.NN = nn.MountainCarNeuronalNetwork(warm_start=warm_start, nbr_neuron_rows=10, nbr_neuron_cols=10, init_weight=0.5)

    def choose_action(self, state):
        return self.NN.choose_action(state)

    def train(self,
              n_steps=2000,
              n_episodes=500,
              learning_rate=0.01,
              reward_factor=0.95,
              eligibility_decay=0.6,
              step_penalty=-0.01,
              tau=0.1):
        print("NN history:", self.NN.history)
        self.NN.show_output(figure_name='start')
        sucess_indexes, traces = self.NN.train(n_steps=n_steps, n_episodes=n_episodes,
                                               learning_rate=learning_rate,
                                               reward_factor=reward_factor,
                                               eligibility_decay=eligibility_decay,
                                               step_penalty=step_penalty,
                                               tau=tau,
                                               save_to_file=True, show_intermediate=False)
        print(self.NN)
        self.NN.show_output(figure_name='last')

        # show learning courve
        plb.figure()
        plb.plot(sucess_indexes, 'o')
        W = max(int(n_episodes/20), 10)
        mean_arr = [n_steps]*W + [np.mean(sucess_indexes[k-W:k]) for k in range(W, len(sucess_indexes))]
        plb.plot(mean_arr, 'r')
        #plb.figure()
        #for t in traces:
        #    plb.plot([s.x for s in t], [s.v for s in t])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Mountain Car with a neuronal network')
    parser.add_argument('-f', dest='filename', required=False, default=None)
    args = parser.parse_args()
    print("args:", args)

    agent = SarsaAgent(warm_start=args.filename)
    agent.train()
    #visualize_trial(agent, n_steps=2000)
    plb.show(block=True)