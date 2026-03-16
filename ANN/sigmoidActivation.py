import matplotlib.pyplot as plt
import numpy as np
# import math

x = np.linspace(-10, 10, 100)
z = 1/(1 + np.exp(-x))

# for each value of x printed corresponding z value
for i in range(100):
    print([x[i], z[i]])


# plt.plot(x,z)
# plt.xlabel("X")
# plt.ylabel("Sigmoid(X)")

# plt.show()
