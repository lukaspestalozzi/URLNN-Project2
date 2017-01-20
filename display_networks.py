from neuronalnetwork import *
import matplotlib.pyplot as plt
from os.path import isdir
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Mountain Car with a neuronal network')
    parser.add_argument(dest='dirname', default="./")
    args = parser.parse_args()
    print("args:", args)

    if isdir(args.dirname):
        networks = [args.dirname+f for f in os.listdir(args.dirname) if '.json' in f]
    else:
        networks = [args.dirname]

    for nf in networks:
        net = MountainCarNeuronalNetwork(warm_start=nf)
        net.display_network(name=nf)
    plt.show(block=True)
