
import mountaincar as mc
import numpy as np
from collections import namedtuple
from collections import defaultdict
import matplotlib.pylab as plb
import matplotlib.pyplot as plt
from time import time
State = namedtuple('State', ['x', 'v'])

class SarsaMountainCar(object):

    def __init__(self, learning_rate=0.1, reward_factor=0.95, eligibility_decay=0.7):
        self.learning_rate = learning_rate
        self.reward_factor = reward_factor
        self.eligibility_decay = eligibility_decay

    

    def _vizualize(self):
        pass
