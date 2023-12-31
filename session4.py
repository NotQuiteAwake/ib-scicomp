# session4.py
# Root-finding in an arbitrary n-th order polynomial with bisection.

import matplotlib.pyplot as plt
from numpy import linspace as lsp

# Define a general polynomial f.
def f(x, n, cof):
    t = 0
    for i in range(n + 1):
        t = t + cof[i] * pow(x, i)
    return t

# Input
print("Finding a root to specified accuracy in an arbitrary polynomial using bisection.")
global N, acc, cof, l, r, vmode

vmode = (input("Verbose mode: output each step? (Y/n): ").lower() == 'y')
if (input("Use a preset sample input? (Y/n): ").lower() == 'y'):
    N = 3
    acc = 0.000001
    cof = [2.5, -3, 5, 1] 
    l = -10
    r = -2.5

    print(f"""----
Using the preset parameters:
Order of polynomial, N = {N}
Required accuracy, acc = {acc}
Coefficients from lowest order, cof = {cof}
Left bound, l = {l}
Right bound, r = {r}""")

else:
    N = int(input("Order of polynomial, N = "))
    acc = float(input("Desired accuracy of root = "))
    print("Now enter coefficient from lowest order (constant) to highest order on the same line, separated by a space.")
    cof = []
    for i in range(N + 1):
        cof.append(float(input(f"x^{i} coefficient = ")))

    # Initial plot
    while (True):
        if (input("(Re)plot the function to put in initial guesses? (Y/n): ").lower() == "y"):
            xl = float(input("Left boundary for plot xl = "))
            xr = float(input("Right boundary for plot xr = "))
            x = lsp(xl, xr, 1000)
            y = f(x, N, cof)
            plt.plot(x, y)
            plt.xlim(xl, xr)
            plt.grid()
            plt.show()
        else:
            l = float(input("Left bound, l = "))
            r = float(input("Right bound, r = "))
            break

print("----")

if not (f(l, N, cof) * f(r, N, cof) < 0):
    print("Poor choice of initial variables. Their product should be negative or no solution is guaranteed.")
else:
    cnt = 0
    while (r - l > acc):
        cnt = cnt + 1
        m = (l + r) / 2
        if (vmode):
            print(f"""Step {cnt}:
    l = {l}, f(l) = {f(l, N, cof)}
    r = {r}, f(r) = {f(r, N, cof)}
    m = {m}, f(m) = {f(m, N, cof)}
    error = {r - l}""")
        if (f(l, N, cof) * f(m, N, cof) < 0):
            r = m
            if (vmode): print("    m -> r.")
        else:
            l = m
            if (vmode): print("    m -> l")
    
    if vmode: print("----")
    print(f"Root x = {(l + r) / 2} determined to within {r - l} after {cnt} steps")
