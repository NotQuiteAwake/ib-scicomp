# session3.py
# Estimation of ln(2) with Monte Carlo algorithm

import matplotlib.pyplot as plt
from random import random
from math import log
import numpy as np

print("Calculates ln(2) with a Monte Carlo algorithm and plot a graph.")
N = int(input("Number of iterations N = "))

# List of results for randomised x and y.
# xs, ys: Coordinates for points below 1/x.
# xf, yf: Coordinates for points above 1/x.
xs = []
ys = []
xf = []
yf = []

# Counter for how many are in the right area.
cnt = 0

def f(x):
    return 1 / x

vmode = (input("Verbose mode, ie print out every result? (Y/n): ").lower() == "y")

for i in range(N):
    x = random() + 1
    y = random()
    
    # success flag
    suc = False

    if ((f(x) > y)):
        # If randomised value y is below f(x) = 1/x then it falls within the desired area.
        cnt = cnt + 1
        suc = True
        
    if (suc):
        xs.append(x)
        ys.append(y)
    else:
        xf.append(x)
        yf.append(y)

    if (vmode):
        print(f"n = {i}, x = {x}, y = {y}" + (", below 1/x" if suc else ", above 1/x"))


# To find the area, use the probability times the total area which is just 1.
res = cnt / N * 1
err = abs((log(2) - res) / log(2))

print(f"----\naccurate result = {log(2)}\nMonte Carlo result = {res}\nrelative error = {err}\n----")

if (input("Plot a graph? (Y/n): ").lower() == "y"):
    if (vmode):
        print("Plot a graph...")
    
    plt.scatter(xs, ys, label="below", s=5)
    plt.scatter(xf, yf, label="above", s=5)
    
    # Plotting the "true" 1/x
    xt = np.linspace(1, 2, 1000)
    yt = 1/xt
    plt.plot(xt, yt, 'k', label="$f(x) = 1 / x$", linewidth=4)
   
    # Some fine tuning.
    plt.legend(loc='upper right', prop={'size': 20})
    plt.xlim(1, 2)
    plt.ylim(0, 1)
    
    plt.show()
