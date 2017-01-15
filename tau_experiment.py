import numpy as np
import matplotlib.pyplot as plt



def softmax(q, tau=1.0):
    return np.exp(q / tau) / np.sum(np.exp(q / tau))

def plot_tau_curve(q, from_x=0.001, to_x=5.0, step=0.001):
    x = []
    while from_x < to_x:
        x.append(from_x)
        from_x += step

    y = [softmax(a, t) for t in x]
    plt.plot(x, y)
    plt.show()


a = np.array([0.3, 0.3, 0.4])
plot_tau_curve(a, to_x=1.0, step=0.0001)
