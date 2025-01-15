#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
data = np.loadtxt('data.txt')
xi, fi = data[:, 0], data[:, 1]

plt.plot(xi, fi, label='f(x) = xe^(-x^2)')
plt.xlabel('xi')
plt.ylabel('f(x)')
plt.title('Plot of f(x) = xe^(-x^2)')
plt.legend()
plt.grid(True)
plt.show()


