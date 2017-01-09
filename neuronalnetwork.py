import numpy as np
from collections import namedtuple
import mountaincar as mc
from collections import defaultdict
import matplotlib.pyplot as plt
from time import time, strftime
import json
import os

class IllegalArgumentError(ValueError):
    pass

State = namedtuple('State', ['x', 'v'])

class MountainCarNeuronalNetwork(object):

    def __init__(self, warm_start=None, nbr_neuron_rows=15, nbr_neuron_cols=15, init_weight=None,
                 tau=0.2,
                 x_min=-150, x_max=5, v_min=-15, v_max=15):
        """
        warm_start: json file name from where to read the state of the NN, if this is not None, all other parameters are ignored

        other parameters are self explanatory
        """

        if warm_start is not None:
            d = self._read_from_file(warm_start)
            self._nbr_neurons_rows = d['nbr_rows']
            self._nbr_neurons_cols = d['nbr_cols']
            self.nbr_neurons = self._nbr_neurons_rows*self._nbr_neurons_cols
            self._tau = d['tau']
            self._x_vals, self._sigma_x = np.array(d['x_vals']), d['sigma_x']
            self._v_vals, self._sigma_v = np.array(d['v_vals']), d['sigma_v']
            self._neurons_w = np.array(d['weights'])
            self.history = d['history']
        else:
            if nbr_neuron_rows < 2 or nbr_neuron_rows < 2:
                raise IllegalArgumentError("nbr_neuron_rows and nbr_neuron_cols must be 2 or bigger!")

            self._nbr_neurons_rows = nbr_neuron_rows
            self._nbr_neurons_cols = nbr_neuron_cols
            self.nbr_neurons = nbr_neuron_cols*nbr_neuron_rows
            self._tau = float(max(tau, 1e-6))
            self._x_vals, self._sigma_x = np.linspace(start=x_min, stop=x_max, num=nbr_neuron_cols, endpoint=False, retstep=True)
            self._v_vals, self._sigma_v = np.linspace(start=v_min, stop=v_max, num=nbr_neuron_rows, endpoint=True, retstep=True)
            self.history = []

            if init_weight is None:
                self._neurons_w = np.random.rand(self.nbr_neurons, 3)
            else:
                self._neurons_w = np.zeros((self.nbr_neurons, 3), dtype=float)
                self._neurons_w.fill(float(init_weight))
        #fi

        self._sigma_x2 = self._sigma_x**2
        self._sigma_v2 = self._sigma_v**2

        self._neurons_e = np.zeros((self.nbr_neurons, 3), dtype=float)
        self._neurons_pos = np.array([(x, v) for x in self._x_vals for v in self._v_vals], dtype=float)

        self.actions = [-1, 0, 1]

        # check some assumptions that must hold
        assert self.nbr_neurons == self._nbr_neurons_rows*self._nbr_neurons_cols
        assert len(self._neurons_w) == len(self._neurons_e) == len(self._neurons_pos) == self.nbr_neurons
        assert self._neurons_w.shape == self._neurons_e.shape == (self.nbr_neurons,3)

    def _read_from_file(self, filename):
        print("reading NN")
        with open(filename, 'r') as f:
            d = json.load(f)
            print(d)
            return d
        return None

    def _store_to_file(self):
        def assure_path_exists(path):
            folder = os.path.dirname(path)
            if not os.path.exists(folder):
                    os.makedirs(folder)
        d = {
            'nbr_rows':self._nbr_neurons_rows,
            'nbr_cols':self._nbr_neurons_cols,
            'tau':self._tau,
            'x_vals':self._x_vals.tolist(),
            'sigma_x':self._sigma_x,
            'v_vals':self._v_vals.tolist(),
            'sigma_v':self._sigma_v,
            'weights':self._neurons_w.tolist(),
            'history':self.history
        }
        print("saving NN:\n", d)
        path = 'networks/r{}_c{}/'.format(self._nbr_neurons_rows, self._nbr_neurons_cols)
        assure_path_exists(path)
        filename = '{}nn_{}.json'.format(path, strftime('%d_%m_%Y-%H:%M:%S'))
        with open(filename, 'w') as f:
            json.dump(d, f)
        print("done saving")

    def _get_Q(self, state, action, activs=None):
        #print(action)
        #assert int(action) == action
        # calculate the activation of all neurons
        if activs is None:
            activs = self._input_neuron_activations(state)

        # weight the activations
        weigths = self._neurons_w[:,action+1]

        mult = weigths*activs
        s = np.sum(mult)
        #print("state:", state, "\na:", activations, "\nw:", weigths, "\nmult:", mult,"\ns:", s)
        #print(["{:.5f}".format(a) for a in activations if a > 0.001])
        #exit()
        #print(activations.shape, weigths.shape, mult.shape, s.shape)
        #assert activations.shape == weigths.shape == mult.shape
        return s

    def _get_Q_all(self, state, activs=None):
        """
        returns the Q values for all 3 actions [-1, 0, 1]
        """
        # calculate the activation of all neurons
        if activs is None:
            activs = self._input_neuron_activations(state)
        #print(in_activs.shape)
        activations = np.reshape(activs, (self.nbr_neurons, 1))

        # weight the activations
        return np.sum(self._neurons_w*activations, axis=0)

    def output_activations(self, state, in_activs=None):
        """
        returns the activity of the output neurons [-1, 0, 1]
        """
        q = self._get_Q_all(state, activs=in_activs)
        denominator = np.sum(np.exp(q / self._tau))
        op_activs = np.exp(q / self._tau) / denominator
        return op_activs


    def choose_action(self, state, in_activs=None):
        """
        Chooses an action based on the exitation of the ouput neurons
        """
        """
        if state.v >= 0:
            return 1
        else:
            return -1
        """
        oput = self.output_activations(state, in_activs=in_activs)
        ret = np.random.choice(self.actions, p=list(oput))
        return ret

    def _decay_E(self, eligibility_decay):
        """
        multiplies the E value of the corresponding neuron by eligibility_decay
        """
        self._neurons_e *= eligibility_decay
        self._neurons_e[self._neurons_e < 0.000001] = 0 # set small values to 0

    def _increment_E(self, state, action, activs=None):
        """
        adds the activation of each neuron to the E value of the neuron
        """
        if activs is None:
            activs = self._input_neuron_activations(state)
        self._neurons_e[:, action+1] += activs
        # truncuate at 1
        self._neurons_e[self._neurons_e > 1] = 1

    def _update_Q(self, state, action, delta_q):
        """
        adds the delta_q to the weights the corresponding neuron and action weighted by the eligibility value
        """
        a_idx = action+1
        #print(state, action, "->", delta_q)
        self._neurons_w[:, a_idx] += self._neurons_e[:, a_idx]*delta_q

    def _reset_E(self):
        self._neurons_e = np.zeros((self.nbr_neurons, 3), dtype=float)


    def _input_neuron_activations(self, state):
        x_diff = (self._neurons_pos[:, 0] - state.x)**2
        v_diff = (self._neurons_pos[:, 1] - state.v)**2
        activs = np.exp(- (x_diff/self._sigma_x2) - (v_diff/self._sigma_v2))
        activs[activs < 1e-6] = 0.0
        return activs

    def train(self, n_steps, n_episodes, learning_rate, reward_factor, eligibility_decay, tau, step_penalty=-0.1, mountain_car=None, save_to_file=True, show_intermediate=False):
        """
        save_to_file: if True, then stores the NN after the training to a file.
        """
        self._tau = tau
        if mountain_car is None:
            mountain_car = mc.MountainCar()

        if n_steps is None:
            n_steps = float('inf')
        sucess_indexes = []
        traces = []
        for ep in range(n_episodes):
            t = time()
            print("episode", ep, "/", n_episodes) #, self.mean_positivenegative_v(delim=" "))
            idx, trace = self._episode(mountain_car, learning_rate=learning_rate, reward_factor=reward_factor, eligibility_decay=eligibility_decay, n_steps=n_steps, step_penalty=step_penalty)
            sucess_indexes.append(idx)
            #traces.append(trace)
            print("  (t={:.4f})".format(time()-t))
            self.show_output(figure_name='activations_interactive', interactive=True)
            if show_intermediate and ep % 100 == 99:
                self.show_output(figure_name='activations_'+str(ep), interactive=False)

        self.history.append({'episodes':n_episodes,
                             'steps':n_steps,
                             'learning_rate':learning_rate,
                             'reward_factor':reward_factor,
                             'eligibility_decay':eligibility_decay,
                             'step_penalty':step_penalty})
        if save_to_file:
            self._store_to_file()
        return sucess_indexes, traces

    def _episode(self, mountain_car, learning_rate, reward_factor, eligibility_decay, n_steps, step_penalty):

        mountain_car.reset(random=True)
        self._reset_E() # set all e to 0

        curr_state = State(mountain_car.x, mountain_car.x_d)
        curr_in_activs = self._input_neuron_activations(curr_state)
        curr_action = self.choose_action(curr_state, in_activs=curr_in_activs)

        trace = [curr_state]
        step = 0
        while step <= n_steps:
            step += 1
            mountain_car.apply_force(curr_action)
            mountain_car.simulate_timesteps(100, 0.01)

            next_state = State(mountain_car.x, mountain_car.x_d)
            next_in_activs = self._input_neuron_activations(next_state)
            r = mountain_car.R
            if r < 1: # penalize if not succeeded
                r = step_penalty

            next_action = self.choose_action(next_state, in_activs=next_in_activs)

            curr_Q = self._get_Q(curr_state, curr_action, activs=curr_in_activs)
            next_Q = self._get_Q(next_state, next_action, activs=next_in_activs)
            delta_q = learning_rate*(r + reward_factor*next_Q - curr_Q)

            self._increment_E(curr_state, curr_action, activs=curr_in_activs)
            self._update_Q(curr_state, curr_action, delta_q)
            self._decay_E(eligibility_decay)
            curr_state = next_state
            curr_action = next_action
            curr_in_activs = next_in_activs
            if mountain_car.R > 0.0:
                print("  -> succeded at step", step)
                return step, trace
        return step, trace

    def show_activations(self, state):
        activs = np.reshape(self._input_neuron_activations(state), (len(self._x_vals), len(self._v_vals)))
        plt.figure()
        plt.imshow(activs, interpolation='gaussian')
        plt.xticks([0, len(self._x_vals)-1], [self._x_vals[0], self._x_vals[-1]])
        plt.yticks([0, len(self._v_vals)-1], [self._v_vals[0], self._v_vals[-1]])
        plt.xlabel("position")
        plt.ylabel("velocity")

    def show_output(self, figure_name, block=False, interactive=False):
        outputs_arr = [self.output_activations(State(n_x, n_v)) for n_x, n_v in self._neurons_pos]
        #print("arr:\n", outputs_arr)
        outputs_matrix = np.zeros((self._nbr_neurons_rows, self._nbr_neurons_cols, 3))
        last_row = self._nbr_neurons_rows-1
        i_x = 0
        i_v = last_row
        for activ in outputs_arr:
            #print(i_v, i_x, "->", activ)
            outputs_matrix[i_v, i_x] = activ
            i_v = (i_v - 1) % self._nbr_neurons_rows
            if i_v == last_row:
                i_x += 1
        #print("matrix:\n", outputs_matrix)

        if interactive:
            plt.ion()
        else:
            plt.ioff()
        plt.figure(figure_name)
        plt.clf()
        plt.imshow(outputs_matrix, interpolation='none')#, interpolation='gaussian')
        plt.xticks(range(len(self._x_vals)), self._x_vals, rotation=90.0)
        plt.yticks(range(len(self._v_vals)), reversed(self._v_vals))
        plt.xlabel("position")
        plt.ylabel("velocity")
        plt.pause(0.00000001)
        plt.show(block=block)



    def mean_positivenegative_v(self, delim="\n"):
        positive_v_w = [self._neurons_w[i, :] for i in range(0, self.nbr_neurons) if self._neurons_pos[i][1] > 0]
        negative_v_w = [self._neurons_w[i, :] for i in range(0, self.nbr_neurons) if self._neurons_pos[i][1] < 0]
        s = "positive_v: {}{}".format(np.mean(positive_v_w, axis=0), delim)
        s += "negative_v: {}{}".format(np.mean(negative_v_w, axis=0), delim)
        return s

    def __str__(self):
        s = "Neural Network:\n"
        for n in range(self.nbr_neurons):
            pos_s = "({:.2f}, {:.2f})".format(float(self._neurons_pos[n, 0]), float(self._neurons_pos[n, 1]))
            w_s = "{}".format([float("{:.2f}".format(w)) for w in self._neurons_w[n]])
            e_s = "{}".format([float("{:.2f}".format(e)) for e in self._neurons_e[n]])
            s += "{}: {} {}\n".format(pos_s, w_s, e_s)

        s += self.mean_positivenegative_v()
        s += "history: "+str(self.history)
        return s
