import matplotlib.pyplot as plt
import numpy as np
a = np.load("./loss_record.npy")
plt.stackplot(list(range(a.shape[1])), a.min(0), a.mean(0) - a.min(0), a.max(0) - a.mean(0))
