import sys

import matplotlib.pylab as plb
import numpy as np
import mountaincar
import random as rnd
import neuronalnetwork as nn
from neuronalnetwork import State
from collections import defaultdict
import argparse

def visualize_trial(agent, n_steps=200):
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
    def __init__(self,warm_start=None):

        self.NN = nn.MountainCarNeuronalNetwork(warm_start=warm_start, nbr_neuron_rows=20, nbr_neuron_cols=20, init_weight=0.5)

    def save_nn(self, directory=None):
        """
        directory: saves the NN into the directory
        """
        self.NN._store_to_file(path=directory)

    def choose_action(self, state):
        return self.NN.choose_action(state)

    def train_and_show(self, n_steps=None, n_episodes=None):
        print("NN history:", self.NN.history)
        self.NN.show_output(figure_name='start', tau=0.5)
        self.train(n_steps, n_episodes)
        print(self.NN)
        self.NN.display_network(name="after training")


    def train(self, n_steps=None, n_episodes=None):
        epis = 300 if n_episodes is None else n_episodes
        return self.NN.train(n_steps=2000 if n_steps is None else n_steps,
                             n_episodes=epis,
                             reward_factor=0.95,
                             eligibility_decay=0.9,
                             step_penalty=-0.0,
                             init_learning_rate=0.15,
                             duration_learingrate=100,
                             target_learning_rate=0.02,
                             min_learning_rate=0.02,
                             init_tau=1.0,
                             duration_tau=150,
                             target_tau=0.01,
                             min_tau=0.01,
                             save_to_file=True,
                             show_intermediate=True,
                             show_trace=False,
                             show_interactive=True,
                             show_weights=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Mountain Car with a neuronal network')
    parser.add_argument('-f', dest='filename', required=False, default=None)
    parser.add_argument('-s', dest='n_steps', required=False, default=None, type=int)
    parser.add_argument('-e', dest='n_episodes', required=False, default=None, type=int)
    args = parser.parse_args()
    print("args:", args)

    agent = SarsaAgent(warm_start=args.filename)
    try:
        agent.train_and_show(n_steps=args.n_steps, n_episodes=args.n_episodes)
        plb.show(block=True)
    except KeyboardInterrupt:
        agent.save_nn()
        print("exiting...")
        exit()
