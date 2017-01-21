from neuronalnetwork import *
import matplotlib.pyplot as plt
from os.path import isdir
import argparse


def smoothTriangle(data,degree,dropVals=False):
    """performs moving triangle smoothing with a variable degree."""
    """note that if dropVals is False, output length will be identical
    to input length, but with copies of data at the flanking regions"""
    triangle = np.array(list(range(degree)) + [degree] + list(range(degree))[::-1]) +1
    smoothed = []
    for i in range(degree,len(data)-degree*2):
        point = data[i:i+len(triangle)]*triangle
        smoothed.append(sum(point)/sum(triangle))
    if dropVals:
        return smoothed
    smoothed = [smoothed[0]]*(degree+degree//2)+smoothed
    while len(smoothed)<len(data):
        smoothed.append(smoothed[-1])
    return smoothed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Mountain Car with a neuronal network')
    parser.add_argument(dest='dirname', default="./")
    args = parser.parse_args()
    print("args:", args)

    if isdir(args.dirname):
        networks = [args.dirname+f for f in os.listdir(args.dirname) if '.json' in f]
    else:
        raise ValueError("the first argument must be a directory")

    all_succ_idxs = []
    for nf in networks:
        net = MountainCarNeuronalNetwork(warm_start=nf)
        # get the whole learningcurve
        succ_idxs = []
        for h in net.history:
            succ_idxs += h['sucess_indexes']
        plt.plot(succ_idxs, 'b.')
        # save it
        all_succ_idxs.append(succ_idxs)

    # average them
    succ_idxs_averaged = np.mean(np.array(all_succ_idxs), axis=0)
    succ_idxs_smoothed = list(succ_idxs_averaged[:3]) + list(smoothTriangle(succ_idxs_averaged, degree=2, dropVals=True)) + list(succ_idxs_averaged[-3:])

    # plot them
    #plt.plot(range(len(succ_idxs_averaged)), succ_idxs_averaged, 'ro', linewidth=2)
    plt.plot(range(len(succ_idxs_smoothed)), succ_idxs_smoothed, 'r-', linewidth=5)
    plt.show(block=True)
