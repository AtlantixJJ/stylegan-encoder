import matplotlib.pyplot as plt
import numpy as np
import glob

files = glob.glob("reconstruction/*.npy")
files.sort()
losses = [np.load(f) for f in files]
losses = np.array(losses)
print(losses.shape)
plt.stackplot(
    list(range(losses.shape[1])),
    losses.min(0),
    losses.mean(0) - losses.min(0),
    losses.max(0) - losses.mean(0))
plt.savefig("temp.png")
plt.close()