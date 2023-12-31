# session1.py
# Euclidean algorithm for GCD

def gcd(a, b):
    while (b != 0):
        t = b
        b = a % b
        a = t
    return a

print("session1.py: Determines the greatest common divisor of two numbers using the Euclidean algorithm.")

a = int(input("First integer, a = "))
b = int(input("Second integer, b = "))

if (a <= 0 or b <= 0):
    print("That won't do. Positive numbers only.")

else:
    res = gcd(a, b)
    print(f"The GCD between {a} and {b} is {res}")
