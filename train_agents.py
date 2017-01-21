from neuronalnetwork import *
import matplotlib.pyplot as plt
from starter import SarsaAgent
from os.path import isdir
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Mountain Car with a neuronal network')
    parser.add_argument(dest='n_agents', type=int)
    parser.add_argument(dest='n_episodes', type=int)
    parser.add_argument('-s', dest='n_steps', required=False, default=2000, type=int)

    args = parser.parse_args()
    print("args:", args)

    for k in range(args.n_agents):
        print("==============================================")
        print("Training {} of {}".format(k+1, args.n_agents))

        a = SarsaAgent(warm_start=None)
        a.train(n_episodes=args.n_episodes)
        a.save_nn(directory='networks/average_10/')

    print("done")
