import numpy as np
import matplotlib.pyplot as plt

lr = 0.15
lrs = [lr]
lrd = 2000
lrf = (0.05 / lr)**(1.0/lrd)

for k in range(0, 4000):
    lr *= lrf
    lrs.append(lr)

plt.plot([max(0.02, l) for l in lrs])
plt.show()
