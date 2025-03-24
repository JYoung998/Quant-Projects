import numpy as np
sims = 1000000

A = np.random.uniform(1, 5, sims)
B = np.random.uniform(1, 5, sims)

duration = A + B 

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.hist(duration, bins=100, color='skyblue', edgecolor='black')
plt.hist(duration, density = True)
plt.axvline(9, color = 'red', linestyle = 'dashed', linewidth = 2)
plt.show()
print((duration > 9).sum() / sims)

