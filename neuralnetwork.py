
# coding: utf-8
import numpy as np
from collections import namedtuple
import mountaincar as mc
from collections import defaultdict
import matplotlib.pylab as plb
import matplotlib.pyplot as plt
from time import time


class IllegalArgumentError(ValueError):
    pass

State = namedtuple('State', ['x', 'v'])
OutputNeuron = namedtuple('OutputNeuron', ['command'])
class InputNeuron(object):

    def __init__(self, x, v, init_weight=1.0):
        """
        x: x position of the neuron
        v: x_dot, or velocity position of the neuron
        init_weight: (default 1) the initial weight of the neuron for each output neuron
                     if it is an integer, then all weight will be this value
                     if it is a list, then it will be mapped to  [-1, 0, 1]
                     if it is None, then a random value is chosen (between 0 and 100)
        """
        self.x = x
        self.v = v
        self.e = [0.0, 0.0, 0.0]
        if init_weight is None:
            self._weights = [np.random.rand() for i in range(0, 3)]
        elif type(init_weight) is int or type(init_weight) is float:
            self._weights = [float(init_weight)]*3
        elif type(init_weight) is list and len(init_weight) == 3:
            self._weights = list(init_weight)
        else:
            raise IllegalArgumentError("init_weight must have length 3! But was "+str(init_weight))

    def activity(self, state, sigma_x, sigma_v):
        ac = np.exp(-((self.x - state.x)**2/sigma_x) - ((self.v - state.v)**2)/sigma_v)
        return ac

    def weight(self, action):
        """
        Returns the weight for the given action. action must be either -1, 0 or 1
        """
        return self._weights[action+1]

    def get_E(self, action):
        return self.e[action+1]

    def update_E(self, action, new_e):
        """
        sets the E to new_e
        """
        if new_e < 1e-6:
            new_e = 0.0
        self.e[action+1] = new_e

    def update_weight(self, action, val):
        """
        adds the val to the weight
        """
        self._weights[action+1] += val

    def dist(self, state):
        return abs(self.x - state.x) + abs(self.v - state.v)

    def __str__(self):
        s = "Neuron(x={}, v={}, e={}, weights={})".format(self.x, self.v, self.e, self._weights)
        return s

class MountainCarneuralNetwork(object):

    def __init__(self, nbr_input_neurons=10, x_min=-150, x_max=30, v_min=-15, v_max=15, init_weight=1.0):
        if nbr_input_neurons < 2 or int(nbr_input_neurons) != nbr_input_neurons:
            raise IllegalArgumentError("nbr_input_neurons must be an integer >= 2.")

        self.x_vals, self.sigma_x = np.linspace(start=x_min, stop=x_max, num=nbr_input_neurons, endpoint=True, retstep=True)
        self.v_vals, self.sigma_v = np.linspace(start=v_min, stop=v_max, num=nbr_input_neurons, endpoint=True, retstep=True)

        self.neurons = [InputNeuron(x, v, init_weight=init_weight) for x in self.x_vals for v in self.v_vals]
        self.actions = [-1, 0, 1]
        self.output_neurons = [OutputNeuron(c) for c in self.actions]

    def get_Q(self, state, action):
        q = np.sum([n.weight(action)*n.activity(state, self.sigma_x, self.sigma_v) for n in self.neurons])
        return q

    def _closest_neuron(self, state):
        idx = np.argmin([n.dist(state) for n in self.neurons])
        return self.neurons[idx]

    def _update_E(self, state, action, new_e):
        neuron = self._closest_neuron(state)
        neuron.update_E(action, neuron.get_E(action) + new_e)

    def output(self, state, tau=1.0):
        """
        Returns a dict {-1: e_-1, 0: e_0, 1: e_1} with e_i as the exitation for each output neuron
        """
        tau = float(tau)
        denominator = np.sum(np.exp(np.array([self.get_Q(state, action) for action in self.actions]) / tau))
        exitation = {}
        if denominator < 1e-6:
            print("denominator < 1e-6")
            return {-1: (1.0/3.0), 0: (1.0/3.0), 1: (1.0/3.0)}

        for action in self.actions:
            nominator = np.exp(self.get_Q(state, action) / tau)
            exitation[action] = nominator / denominator

        return exitation

    def choose_action(self, state, tau=1.0):
        """
        Chooses an action based on the exitation of the ouput neurons
        """
        oput = self.output(state, tau=tau)
        ret = np.random.choice(self.actions, p=list(oput.values()))
        return ret

    def train(self, mountain_car=None, n_steps=200, max_nbr_episodes=2,
              reward_factor=0.9, learning_rate=0.5, eligibility_decay=0.6, visualize=False):
        """
        Learning with Sarsa Algoritm
        """
        def prepare_viz():
            mv = None
            if visualize:
                plb.ion()
                mv = mc.MountainCarViewer(mountain_car)
                mv.create_figure(n_steps=n_steps, max_time=n_steps)
                plb.draw()
                plb.pause(0.0001)
            return mv

        def update_viz(mv):
            if visualize:
                mv.update_figure()
                plb.draw()
                plb.pause(0.0001)


        print("training...")
        if mountain_car is None:
            mountain_car = mc.MountainCar()

        if n_steps is None:
            n_steps = float('inf')

        success_steps = []
        for episode in range(0, max_nbr_episodes):
            print("episode", episode, "/", max_nbr_episodes)
            mountain_car.reset()
            mv = prepare_viz()
            curr_state = State(mountain_car.x, mountain_car.x_d)
            curr_action = self.choose_action(curr_state)
            #print("init state: ", curr_state)
            t = time()
            step = 0
            while step < n_steps:
                step += 1
                mountain_car.apply_force(curr_action)
                mountain_car.simulate_timesteps(100, 0.01)
                update_viz(mv)
                new_state = State(mountain_car.x, mountain_car.x_d)
                r = mountain_car.R
                new_action = self.choose_action(new_state)
                delta = learning_rate*(r + reward_factor*self.get_Q(new_state, new_action) - self.get_Q(curr_state, curr_action))
                #print("delta", delta)
                self._update_E(curr_state, curr_action, 1)
                for neuron in self.neurons:
                    for a in self.actions:
                        ne = neuron.get_E(a)
                        if ne > 0.0:
                            val_q = delta*ne
                            neuron.update_weight(a, val_q)
                            val_e = eligibility_decay*reward_factor*ne
                            neuron.update_E(a, val_e)
                curr_state = new_state
                curr_action = new_action
                if mountain_car.R > 0.0:
                    print("succeded at step", step, end=' ')
                    success_steps.append(step)
                    break

            print("(time: "+str(time()-t)+")")
            if step >= n_steps:
                success_steps.append(step)
        #print("storing caches...", end="")
        #mountain_car.store_caches()
        #print("...done")
        return success_steps

    def __str__(self):
        w = [str(n) for n in self.neurons]
        return '\n'.join(w)

    def show_output(self, x_min=-150, x_max=30, v_min=-15, v_max=15, granularity=30):
        print("plotting...")
        arr_output = np.zeros((len(self.x_vals), len(self.v_vals), 3))
        arr_output = np.zeros((len(self.x_vals), len(self.v_vals), 3))
        v_vals_rev = list(reversed(self.v_vals))
        for ix, x in enumerate(self.x_vals):
            print(".", end="")
            for iv, v in enumerate(v_vals_rev):
                o0, o1, o_1 = self.output(State(x, v)).values()
                arr_output[ix][iv][0] = o_1 # r
                arr_output[ix][iv][1] = o0  # g
                arr_output[ix][iv][2] = o1  # b

        plt.figure()
        plt.imshow(arr_output, interpolation='gaussian')
        plt.xticks([0, len(self.x_vals)-1], [self.x_vals[0], self.x_vals[-1]])
        plt.yticks([0, len(v_vals_rev)-1], [v_vals_rev[0], v_vals_rev[-1]])
        plt.xlabel("position")
        plt.ylabel("velocity")
        print("... done plotting")
