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

    def __init__(self, warm_start=None, nbr_neuron_rows=10, nbr_neuron_cols=15, init_weight=None,
                 x_min=-150, x_max=5, v_min=-15, v_max=15,
                 cheat=False, print_nn=True):
        """
        warm_start: json file name from where to read the state of the NN, if this is not None, all other parameters are ignored
        cheat: if True, initial weights are set such that the car always accelerates in the direction it travels. (default False)

        other parameters are self explanatory
        """

        if warm_start is not None:
            d = self._read_from_file(warm_start)
            self._nbr_neurons_rows = d['nbr_rows']
            self._nbr_neurons_cols = d['nbr_cols']
            self.nbr_neurons = self._nbr_neurons_rows*self._nbr_neurons_cols
            self._x_vals, self._sigma_x = np.array(d['x_vals']), d['sigma_x']
            self._v_vals, self._sigma_v = np.array(d['v_vals']), d['sigma_v']
            self._neurons_w = np.array(d['weights'])
            self.history = d['history']
        else:
            if nbr_neuron_rows < 2 or nbr_neuron_cols < 2:
                raise IllegalArgumentError("nbr_neuron_rows and nbr_neuron_cols must be 2 or bigger!")

            self._nbr_neurons_rows = nbr_neuron_rows
            self._nbr_neurons_cols = nbr_neuron_cols
            self.nbr_neurons = nbr_neuron_cols*nbr_neuron_rows

            self._x_vals, self._sigma_x = np.linspace(start=x_min, stop=x_max, num=nbr_neuron_cols, endpoint=True, retstep=True)
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

        # cheating:
        if cheat:
            self._neurons_w.fill(0.0)
            for i in range(0, self.nbr_neurons):
                if self._neurons_pos[i, 1] >= 0:
                    self._neurons_w[i, 2] = 1.0
                else:
                    self._neurons_w[i, 0] = 1.0

        # for the plots
        self._x_ticks =(range(len(self._x_vals)), [round(x, 1) for x in self._x_vals])
        self._y_ticks = (range(len(self._v_vals)), [round(v, 1) for v in reversed(self._v_vals)])

        # check some assumptions that must hold
        assert self.nbr_neurons == self._nbr_neurons_rows*self._nbr_neurons_cols
        assert len(self._neurons_w) == len(self._neurons_e) == len(self._neurons_pos) == self.nbr_neurons
        assert self._neurons_w.shape == self._neurons_e.shape == (self.nbr_neurons,3)

        # print the network
        print(self.__str__())

    def _read_from_file(self, filename):
        print("reading NN from "+filename)
        with open(filename, 'r') as f:
            d = json.load(f)
            return d
        return None

    def _store_to_file(self, path=None):
        def assure_path_exists(path):
            folder = os.path.dirname(path)
            if not os.path.exists(folder):
                    os.makedirs(folder)
        d = {
            'nbr_rows':self._nbr_neurons_rows,
            'nbr_cols':self._nbr_neurons_cols,
            'x_vals':self._x_vals.tolist(),
            'sigma_x':self._sigma_x,
            'v_vals':self._v_vals.tolist(),
            'sigma_v':self._sigma_v,
            'weights':self._neurons_w.tolist(),
            'history':self.history
        }
        if path is None:
            path = 'networks/r{}_c{}_e{}_sp{}/'.format(self._nbr_neurons_rows, self._nbr_neurons_cols, self.history[-1]['eligibility_decay'], self.history[-1]['step_penalty'])
        assure_path_exists(path)
        # calculate how many epoches it was trained for
        nepochs = 0
        for h in self.history:
            nepochs += len(h['sucess_indexes'])
        filename = '{}{}_{}.json'.format(path,strftime('%d_%m_%Y-%H:%M:%S'), nepochs)
        with open(filename, 'w') as f:
            #print("saving NN to "+filename, end="")
            json.dump(d, f)
            print("...done saving")

    def _get_Q(self, state, action, activs=None):
        # calculate the activation of all neurons
        if activs is None:
            activs = self._input_neuron_activations(state)

        # weight the activations
        weigths = self._neurons_w[:,action+1]

        mult = weigths*activs
        s = np.sum(mult)
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
        q = np.sum(self._neurons_w*activations, axis=0)
        return q

    def output_activations(self, state, tau, in_activs=None):
        """
        returns the activity of the output neurons [-1, 0, 1]
        """
        if in_activs is None:
            in_activs = self._input_neuron_activations(state)

        q = self._get_Q_all(state, activs=in_activs)
        denominator = np.sum(np.exp(q / tau))

        if denominator == np.inf or denominator == np.nan or denominator <= 0.0: # handle nummeric overflow
            #print("(info: overflow occured)")
            return self._greedy_output(q)

        op_activs = np.exp(q / tau) / denominator
        if np.inf in op_activs or np.nan in op_activs: # handle nummeric overflow
            #print("(info: overflow occured)")
            return self._greedy_output(q)
        else:
            return op_activs

    def _greedy_output(self, q):
        """
        q: array of length 3
        code:
        op_activs = np.zeros(3)
        op_activs[np.argmax(q)] = 1.0
        return op_activs
        """
        op_activs = np.zeros(3)
        op_activs[np.argmax(q)] = 1.0
        return op_activs


    def choose_action(self, state, tau, in_activs=None):
        """
        Chooses an action based on the exitation of the ouput neurons
        """
        if in_activs is None:
            in_activs = self._input_neuron_activations(state)
        oput = self.output_activations(state, in_activs=in_activs, tau=tau)
        ret = np.random.choice(self.actions, p=list(oput))
        return ret

    def _decay_E(self, eligibility_decay):
        """
        multiplies the E value of the corresponding neuron by eligibility_decay
        """
        self._neurons_e *= eligibility_decay
        self._neurons_e[self._neurons_e < 0.000001] = 0.0 # set small values to 0

    def _increment_E(self, state, action, activs=None):
        """
        adds the activation of each neuron to the E value of the neuron
        """
        if activs is None:
            activs = self._input_neuron_activations(state)
        self._neurons_e[:, action+1] += activs
        # truncuate at 1
        self._neurons_e[self._neurons_e > 1] = 1


    def _reset_E(self):
        self._neurons_e = np.zeros((self.nbr_neurons, 3), dtype=float)


    def _input_neuron_activations(self, state):
        x_diff = (self._neurons_pos[:, 0] - state.x)**2
        v_diff = (self._neurons_pos[:, 1] - state.v)**2
        activs = np.exp(- (x_diff/self._sigma_x2) - (v_diff/self._sigma_v2))
        activs[activs < 1e-6] = 0.0
        return activs

    def train(self, n_steps, n_episodes, reward_factor, eligibility_decay,
              init_learning_rate, duration_learingrate, target_learning_rate,
              init_tau, duration_tau, target_tau,
              min_learning_rate=0.005,
              min_tau=0.01, # must not be lower than 0.01
              step_penalty=-0.1, mountain_car=None,
              save_to_file=True,
              show_intermediate=False,
              show_trace=False, show_interactive=True, show_weights=False):
        """
        duration_*: positive integer. Determines at which episode the * parameter reaches it's minimum value. Note that the parameter continues to shrink when it reached the target_learning_rate value.
        min_*: spezifies a lower bound on the * parameter
        save_to_file: if True, then stores the NN after the training to a file. can be a string (directory where to store the NN)
        show_intermediate: if True, shows a plot all 100 episodes
        show_trace: if True, shows the trace of the car for each episode
        """

        #parameter checks
        assert init_tau > 0.0
        assert init_learning_rate != 0.0
        assert n_steps is None or n_steps > 0
        assert n_episodes > 0
        assert duration_tau > 0
        assert duration_learingrate > 0



        tau = float(init_tau)
        learning_rate = float(init_learning_rate)
        tau_update_factor = (target_tau / init_tau)**(1.0/duration_tau)
        learning_rate_update_factor = (target_learning_rate / init_learning_rate)**(1.0/duration_learingrate)

        if mountain_car is None:
            mountain_car = mc.MountainCar()

        if n_steps is None:
            n_steps = float('inf')

        # init history
        self.history.append({ 'episodes' :n_episodes,
                              'steps' :n_steps,
                              'init_learning_rate':init_learning_rate, 'duration_learingrate':duration_learingrate, 'target_learning_rate':target_learning_rate, "min_learning_rate":min_learning_rate,
                              'init_tau':init_tau, 'duration_tau':duration_tau, 'target_tau':target_tau, "min_tau":min_tau,
                              'eligibility_decay' :eligibility_decay,
                              'step_penalty' :step_penalty,
                              'reward_factor':reward_factor,
                              'eligibility_decay':eligibility_decay,
                              'step_penalty':step_penalty,
                              'sucess_indexes' :[],
                              })

        for ep in range(n_episodes):
            # run episode
            t = time()
            print("episode", ep, "/", n_episodes, "tau:", tau, "lrate:", learning_rate)
            idx, trace = self._episode(mountain_car, learning_rate=learning_rate, reward_factor=reward_factor,
                                       eligibility_decay=eligibility_decay, n_steps=n_steps, step_penalty=step_penalty, tau=tau)
            self.history[-1]['sucess_indexes'].append(idx)
            print("  calc_t={:.4f}s".format(time()-t))

            # update tau and learning rate
            tau = max(min_tau, tau*tau_update_factor)
            learning_rate = max(min_learning_rate, learning_rate*learning_rate_update_factor)
            #learning_rate = max(min_learning_rate, 1.0/np.sqrt(ep+44)) # sqrt



            t = time()
            # show some stuff
            if show_interactive:
                self.show_output(figure_name='activations_interactive', tau=tau, interactive=True)
                self.show_vector_field(figure_name='vector field interactive', tau=tau, interactive=True)
            if show_trace is True or (show_trace == 'not_succeeded' and idx > n_steps-2):
                self.show_trace(figure_name='trace_interactive', trace=trace, interactive=True)
            if show_intermediate and ep % 100 == 99:
                self.show_output(figure_name='activations_'+str(ep), tau=tau, interactive=False)
                self.show_vector_field(figure_name='vector field'+str(ep), tau=tau, interactive=False)
            if show_weights is True:
                self.show_weights(figure_name="weights", interactive=True)
            if show_weights == 'intermediate' and ep % 1000 == 999:
                self.show_weights(figure_name="weights_"+str(ep), interactive=False)

            print("  plot_t={:.4f}s".format(time()-t))
        #end for episodes

        # save the NN
        if save_to_file is True:
            self._store_to_file()
        elif isinstance(save_to_file, str):
            self._store_to_file(path=save_to_file)

        # concatenate all previous success_indexes
        ret_si = []
        for h in self.history:
            ret_si += h['sucess_indexes']
        return ret_si

    def _episode(self, mountain_car, learning_rate, reward_factor, eligibility_decay, n_steps, step_penalty, tau):

        mountain_car.reset()
        self._reset_E() # set all e to 0

        curr_state = State(mountain_car.x, mountain_car.x_d)
        curr_in_activs = self._input_neuron_activations(curr_state)
        curr_action = self.choose_action(curr_state, tau, in_activs=curr_in_activs)

        trace = [curr_state] # stores all the states the car was in during the episode
        step = 0
        while step <= n_steps:
            step += 1
            mountain_car.apply_force(curr_action)
            mountain_car.simulate_timesteps(100, 0.01)

            next_state = State(mountain_car.x, mountain_car.x_d)
            next_in_activs = self._input_neuron_activations(next_state)
            r = step_penalty
            if mountain_car.R > 0: # if there is a reward
                r = mountain_car.R

            next_action = self.choose_action(next_state, tau, in_activs=next_in_activs)

            curr_Q = self._get_Q(curr_state, curr_action, activs=curr_in_activs)
            next_Q = self._get_Q(next_state, next_action, activs=next_in_activs)
            delta_q =  r + reward_factor*next_Q - curr_Q  #learning_rate*(r + reward_factor*next_Q - curr_Q)

            # update e
            #self._increment_E(curr_state, curr_action, activs=curr_in_activs)
            self._neurons_e[:, curr_action+1] += curr_in_activs

            #update the weights
            self._neurons_w += self._neurons_e*learning_rate*delta_q

            #decay e
            #self._decay_E(eligibility_decay)
            self._neurons_e *= eligibility_decay*reward_factor
            self._neurons_e[self._neurons_e > 1] = 1.0 # truncuate at 1
            self._neurons_e[self._neurons_e < 0.000001] = 0.0 # set small values to 0


            curr_state = next_state
            curr_action = next_action
            curr_in_activs = next_in_activs
            trace.append(curr_state)
            if mountain_car.R > 0.0:
                print("  -> succeded at step", step)
                return step, trace
        return step, trace

    def show_activations(self, state, activs=None, block=False):
        """
        Shows the activations of the input neurons for a given state
        """
        if activs is None:
            activs = self._input_neuron_activations(state)
        # reshape and rotate by 90
        activs_matrix = np.rot90(activs.reshape((self._nbr_neurons_cols, self._nbr_neurons_rows)), 1)

        plt.ion()
        plt.figure("activation")
        plt.clf()
        plt.imshow(activs_matrix, interpolation='nearest')
        plt.xticks(self._x_ticks[0], self._x_ticks[1], rotation=90.0)
        plt.yticks(self._y_ticks[0], self._y_ticks[1])
        plt.xlabel("position")
        plt.ylabel("velocity")
        plt.show(block=block)
        plt.pause(0.00000001)

    def show_output(self, figure_name, tau, block=False, interactive=False):
        """
        Shows the output neurons (ie. the actions) for each state
        """
        outputs_arr = np.array([self.output_activations(State(n_x, n_v), tau) for n_x, n_v in self._neurons_pos])

        # reshape and rotate by 90
        outputs_matrix = np.rot90(outputs_arr.reshape((self._nbr_neurons_cols, self._nbr_neurons_rows, 3)), 1)
        if interactive:
            plt.ion()
        else:
            plt.ioff()
        plt.figure(figure_name)
        plt.clf()
        plt.title("output neuron activations (tau="+str(tau)+")")
        outputs_matrix[:,:,[0,1,2]] = outputs_matrix[:,:,[2,1,0]]
        plt.imshow(outputs_matrix, interpolation='nearest')#, interpolation='gaussian')
        plt.xticks(self._x_ticks[0], self._x_ticks[1], rotation=90.0)
        plt.yticks(self._y_ticks[0], self._y_ticks[1])
        plt.xlabel("position")
        plt.ylabel("velocity")
        plt.show(block=block)
        plt.pause(0.00000001)

    def show_vector_field(self, figure_name, tau, block=False, interactive=False):
        """
        Shows the vector fields of the activations on each neurons of the grid.
        """
        #Get the actions
        outputs_arr = np.array([self.output_activations(State(n_x, n_v), tau) for n_x, n_v in self._neurons_pos])
        # reshape and rotate by 90
        outputs_matrix = np.rot90(np.fliplr(outputs_arr.reshape((self._nbr_neurons_cols, self._nbr_neurons_rows, 3))),1)

        max_q_indices = np.argmax(outputs_matrix, axis=2)
        max_q = np.max(outputs_matrix, axis=2)
        vectors = np.copy(max_q_indices)
        vectors[vectors == 0] = -1
        vectors[vectors == 1] = 0
        vectors[vectors == 2] = 1
        color = max_q*vectors

        if interactive:
            plt.ion()
        else:
            plt.ioff()
        plt.figure(figure_name)
        plt.clf()
        plt.title("Vector field of Activations and Q values (tau="+str(tau)+")")
        plt.quiver(color, np.zeros((self._nbr_neurons_rows, self._nbr_neurons_cols)), color, units='x', scale=1, scale_units='x')
        #cbar = plt.colorbar()
        #cbar.set_label('Q Value * direction', rotation=270)
        plt.xticks(self._x_ticks[0], self._x_ticks[1], rotation=90.0)
        plt.yticks(self._y_ticks[0], self._y_ticks[1])
        plt.xlabel("position")
        plt.ylabel("velocity")
        plt.show(block=block)
        plt.pause(0.00000001)

    def show_weights(self, figure_name, block=False, interactive=False):
        """
        Shows the highest weight of each input neuron
        """

        if interactive:
            plt.ion()
        else:
            plt.ioff()
        plt.figure(figure_name)
        plt.clf()

        vmin = np.min(self._neurons_w)
        vmax = np.max(self._neurons_w)

        # action weights
        for a in range(3):
            plt.subplot(2, 2, a+1)
            plt.title(str(a-1))
            weights = self._neurons_w[:, a]
            weights_matrix = np.rot90(weights.reshape((self._nbr_neurons_cols, self._nbr_neurons_rows)), 1)
            plt.imshow(weights_matrix, interpolation='nearest', vmin=vmin, vmax=vmax)
            plt.xticks(self._x_ticks[0], self._x_ticks[1], rotation=90.0)
            plt.yticks(self._y_ticks[0], self._y_ticks[1])
            plt.xlabel("position")
            plt.ylabel("velocity")

        # max weights
        plt.subplot(2, 2, 4)
        plt.title("max")
        weights = weights = np.max(self._neurons_w, axis=1)
        weights_matrix = np.rot90(weights.reshape((self._nbr_neurons_cols, self._nbr_neurons_rows)), 1)
        plt.imshow(weights_matrix, interpolation='nearest', vmin=vmin, vmax=vmax)
        plt.xticks(self._x_ticks[0], self._x_ticks[1], rotation=90.0)
        plt.yticks(self._y_ticks[0], self._y_ticks[1])
        plt.xlabel("position")
        plt.ylabel("velocity")


        plt.show(block=block)
        plt.pause(0.00000001)

    def show_trace(self, figure_name, trace, block=False, interactive=False):
        """
        shows the states the car was in.
        trace: must be a list of tuples containing the x and x_dot coordinates. [(-75.0, 3.4), (...,...), ...]
        """
        if interactive:
            plt.ion()
        else:
            plt.ioff()
        plt.figure(figure_name)
        plt.clf()
        plt.plot([s.x for s in trace], [s.v for s in trace],c="green")
        plt.plot(trace[0].x, trace[0].v, 'ro', linewidth=2, c="red") #start
        plt.plot(trace[-1].x, trace[-1].v, 'bo', linewidth=2, c="blue") #end
        plt.xlim(self._x_vals[0],self._x_vals[-1])
        plt.ylim(self._v_vals[0],self._v_vals[-1])
        plt.show(block=block)
        plt.pause(0.00000001)

    def show_learningcurve(self, figure_name, block=False):
        """
        shows the learning curve of this network
        """
        plt.figure(figure_name)
        plt.clf()

        sucess_indexes = []
        for h in self.history:
            sucess_indexes += h['sucess_indexes']

        sucess_indexes = [min(2000, i) for i in sucess_indexes]
        plt.plot(sucess_indexes, 'o')
        W = 5
        mean_arr = [np.mean(sucess_indexes[max(0, k-W):k]) for k in range(1, len(sucess_indexes))]
        plt.plot(range(1, len(mean_arr)+1), mean_arr, 'r', linewidth=2)

        plt.xlabel("epoche")
        plt.ylabel("steps until succeded")
        plt.show(block=block)
        plt.pause(0.00000001)

    def show_parameter_history(self, figure_name, block=False):
        """
        shows the parameter history of this network
        """
        def _exponential_decay(start_val, end_val, duration, min_val, array_length):
            factor = (end_val / start_val)**(1.0/duration)
            val = start_val
            arr = [val]
            for k in range(0, array_length):
                val *= factor
                if val < min_val:
                    nbr_left = array_length - len(arr)
                    arr += [min_val]*nbr_left
                    return arr
                arr.append(val)
            return arr

        plt.figure(figure_name)
        plt.clf()
        # parameters
        #   learning rate
        lrs = []
        #   tau
        taus = []
        #   reward_factor
        rfs = []
        #   eligibility_decay
        eds = []
        # step_penalty
        sps = []

        # vertical lines to show training changes
        vls = []

        for h in self.history:
            #   learning rate
            lrs += _exponential_decay(h["init_learning_rate"], h["target_learning_rate"], h["duration_learingrate"], h["min_learning_rate"], h["episodes"])
            #   tau
            taus += _exponential_decay(h["init_tau"], h["target_tau"], h["duration_tau"], h["min_tau"], h["episodes"])
            #   reward_factor
            rfs += [h["reward_factor"]]*h["episodes"]
            #   eligibility_decay
            eds += [h["eligibility_decay"]]*h["episodes"]
            # step_penalty
            sps += [h["step_penalty"]]*h["episodes"]

            # vertical lines to show training changes
            if len(vls) > 0:
                vls.append(h["episodes"]+vls[-1])
            else:
                vls.append(h["episodes"])



        plt.figure(figure_name)
        rows = 3
        cols = 2
        k = 1
        #   learning rate
        plt.subplot(rows, cols, k)
        plt.title("learning rate")
        plt.plot(lrs)
        plt.vlines(vls, plt.ylim()[0], plt.ylim()[1], colors='r', linestyles='dashdot')
        k += 1

        #   tau
        plt.subplot(rows, cols, k)
        plt.title("tau")
        plt.plot(taus)
        plt.vlines(vls, plt.ylim()[0], plt.ylim()[1], colors='r', linestyles='dashdot')
        k += 1

        #   reward_factor
        plt.subplot(rows, cols, k)
        plt.title("reward_factor")
        plt.plot(rfs)
        plt.vlines(vls, plt.ylim()[0], plt.ylim()[1], colors='r', linestyles='dashdot')
        k += 1

        #   eligibility_decay
        plt.subplot(rows, cols, k)
        plt.title("eligibility_decay")
        plt.plot(eds)
        plt.vlines(vls, plt.ylim()[0], plt.ylim()[1], colors='r', linestyles='dashdot')
        k += 1

        # step_penalty
        plt.subplot(rows, cols, k)
        plt.title("step_penalty")
        plt.plot(sps)
        plt.vlines(vls, plt.ylim()[0], plt.ylim()[1], colors='r', linestyles='dashdot')
        k += 1

        plt.show(block=block)
        plt.pause(0.00000001)

    def display_network(self, name=None, block=False):

        if name is None:
            name = "r{}_c{}_{}".format(self._nbr_neurons_rows, self._nbr_neurons_cols, strftime('%d_%m_%Y-%H:%M:%S'))

        font = {'size'   : 22}

        plt.rc('font', **font)

        # weights
        self.show_weights(figure_name="weights:"+name)

        # output (actions)
        last_tau = max(self.history[-1]['target_tau'], self.history[-1]["min_tau"])
        self.show_output(figure_name="output_tau="+str(last_tau)+":"+name, tau=last_tau)
        self.show_output(figure_name="output_tau=0.5:"+name, tau=0.5)

        # vector field
        self.show_vector_field(figure_name="vector field", tau=last_tau)

        # learning curve
        self.show_learningcurve(figure_name="learningcurve:"+name)

        # parameters
        self.show_parameter_history(figure_name="parameters:"+name)

        # show it
        plt.show(block=block)
        plt.pause(0.00000001)




    def mean_positivenegative_v(self, delim="\n"):
        """
        returns a string containing the average output of the negative and positive velocity neurons separated
        """
        positive_v_w = [self._neurons_w[i, :] for i in range(0, self.nbr_neurons) if self._neurons_pos[i][1] > 0]
        negative_v_w = [self._neurons_w[i, :] for i in range(0, self.nbr_neurons) if self._neurons_pos[i][1] < 0]
        s = "positive_v: {}{}".format(np.mean(positive_v_w, axis=0), delim)
        s += "negative_v: {}{}".format(np.mean(negative_v_w, axis=0), delim)
        return s

    def __str__(self, history=True):
        s = "Neural Network:\n"
        for n in range(self.nbr_neurons):
            pos_s = "({:.2f}, {:.2f})".format(float(self._neurons_pos[n, 0]), float(self._neurons_pos[n, 1]))
            w_s = "{}".format([float("{:.2f}".format(w)) for w in self._neurons_w[n]])
            e_s = "{}".format([float("{:.2f}".format(e)) for e in self._neurons_e[n]])
            s += "{}: {} {}\n".format(pos_s, w_s, e_s)


        s += "sigma_x: {}, sigma_v: {}\n".format(self._sigma_x, self._sigma_v)
        s += self.mean_positivenegative_v()

        if history:
            s += "history: "+str(self.history)
        return s
