# session2.py
# Series expansion of sin about x = 0

from math import factorial as fact
from math import sin, pi

# cof(n): Returns the n-th term in this series
def term(x, n):
    m = 2 * n - 1
    t = pow(x, m) / fact(m)
    if (n % 2 == 0): return -t
    return t


print("Evaluate sin(x) using a Taylor series until a desired accuracy is reached.")

print("x is where sin will be evaluated. Please put in x between -pi and pi.")
x = float(input("x = "))
while not (-pi < x < pi):
    print(f"Please put in an x between -pi and pi.")
    x = float(input("x = "))

print("Accuracy is diff between series result and math.sin(x).")
acc = float(input("accuracy = "))

RES = sin(x)
n = 1
taylor = term(x, 1)
err = abs(taylor - RES)
print(f"----\nn = {n}, series = {taylor}, error = {err}")

while (err > acc):
    n = n + 1
    taylor += term(x, n)
    err = abs(taylor - RES)
    print(f"n = {n}, series = {taylor}, error = {err}")

print(f"----\nDesired accuracy reached with {n} terms. \nsin(x) = {RES}, series = {taylor}, error = {err}")
