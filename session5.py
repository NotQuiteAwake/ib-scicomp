# session5.py
# Simulation of planetary motion

from math import sqrt
from numpy import cross, array, dot
from copy import deepcopy
import matplotlib.pyplot as plt

G = 1 # gravitational constant = 6.6743e-11. Normalise to 1 for our toy model.
D = 3 # dimension to solve in.
ORIGIN = [0] * D # origin vector.

# Astronomical object.
class obj():
    x: array # position vector.
    v: array # velocity vector.
    m: float # mass of object. m = -1 gives dummy object.
    fixed: bool = False # A fixed object will not change its x, regardless of initial velocity. 
    name: str = "" # name of object (optional). In principle there shouldn't be objects with the same name.
   
    def __init__(self, x: array, v: array, m: float, fixed: bool = False, name: str = ""):
        self.x = x
        self.v = v
        self.m = m
        self.fixed = fixed
        self.name = name

"""
List of astronomical objects is stored in a state list.
My convention is such that when a state is passed into any function, they will not be altered, but where they should be updated a new state is returned.
"""

""" add_obj()
Wrapper around object constructor so that x, v are parsed as np.array. One less nightmare.
input: state s, x, v, m
output: new state s2
"""
def add_obj(s: list, x: list, v: list, m: float, fixed: bool = False, name: str = "") -> list:
    s2 = deepcopy(s)
    s2.append(obj(x = array(x), v = array(v), m = m, fixed = fixed, name = name))
    return s2


""" dist()
input: x1[D], x2[D]
output: distance between them. If x1 only then return its length.
"""
def dist(x1: array, x2: array = ORIGIN) -> float:
    dx = x1 - x2
    return sqrt(dot(dx, dx))


""" get_ke()
input: state s, n, index of the object of interest
output: ke of planet n, if n = -1 then total ke.
"""
def get_ke(s: list, n: int = -1) -> float:
    # total ke
    num = len(s)
    if (n == -1):
        e = 0
        for i in range(num):
            e = e + get_ke(s, i)
        return e
    
    # individual ke
    e = 0
    o = s[n]
    return 1/2 * o.m * dot(o.v, o.v)


# similar to get_ke but for pe.
# note that individual pe only really makes sense for a single planet-sun system. 
def get_pe(s: list, n: int = -1) -> float:
    num = len(s)
    if (n == -1):
        e = 0
        for i in range(num):
            e = e + get_pe(s, i)
        return e / 2 # removes double-counting of interactions.
    
    e = 0
    o = s[n]
    for i in range(num):
        if (i == n):
            continue
        o2 = s[i]
        e = e + (-G * o.m * o2.m / dist(o.x, o2.x))
    return e


# similar to get_ke but for total energy.
def get_e(s: list, n: int = -1) -> float:
    return get_ke(s, n) + get_pe(s, n)


"""get_j()
angular momentum stuff.
input: state s, n, index of object, ori, origin to take j from. Default to ORIGIN.
output: j of object n, a vector of 3 dimensions; if n missing then total.
"""
def get_j(s: list, n: int = -1, ori: list = ORIGIN) -> array:
    num = len(s)
    if (n == -1):
        j = array([0] * 3) # Must be done in 3D regardless of D.
        for i in range(num):
            j = j + get_j(s, i)
        return j
    o = s[n]
    
    res = cross(o.x - ori, o.m * o.v)
    # numpy.cross is a funny function. This type of unpredictable behaviour is part of what I hate about python. 
    if (D == 2):
        res = [0, 0, res]
    return array(res)


"""gravity(s, n, m)
input: s, state; n, m, index of objects.
output: force (list[D]) on n due to m. For m missing, total force on n.
"""
def gravity(s: list, n: int, m: int = -1) -> array:
    num = len(s)
    if (m == -1):
        f = ORIGIN
        for i in range(num):
            if (i == n):
                continue
            f = f + gravity(s, n, i)
            return f
    o = s[n]
    o2 = s[m]
    return -G * o.m * o2.m / (dist(o.x, o2.x) ** 3) * (o.x - o2.x)

""" plot_helper
First plot a projected trajectory in XY, then
Plot graphs showing change in energy and angular momentum, as well as x component against time (useful assuming sun fixed at (0, 0))
inputs: 
hist, the full history: a list of states during the simulation.
st, state times, the list of times that the states in history correspond to.
n: Particle to view trajectory / x component.
normal: normal used for calculating angular momentum component. Default to 1z as this works for 2D problems.
"""
def plot_helper(hist: list, st: list, n: int, normal: array = array([0, 0, 1])):
    # First plot: projected trajectory in XY plane. 
    x = [s[n].x for s in hist]
    plt.plot([t[0] for t in x], [t[1] for t in x])
    plt.show()
    
    # For this we need to extract necessary information from hist.
    u = [] # energy
    j = [] # angular momentum in normal.
    x = [] # x-component of nth object, NOT a position vector.
    for s in hist:
        u.append(get_e(s))
        j.append(dot(get_j(s), normal))
        x.append(s[n].x[0])
    plt.close()

    fig, ax = plt.subplots(3)
    fig.suptitle(f'E, J along normal and x of obj {n}')
    ax[0].plot(st, u)
    ax[1].plot(st, j)
    ax[2].plot(st, x)
    for axis in ax:
        axis.axis(xmin = min(st), xmax = max(st))
    ax[0].axis(ymin = min(u) * 1.2, ymax = max(u) * 0.8) # u is negative...
    ax[1].axis(ymin = min(j) * 0.8, ymax = max(j) * 1.2)
    plt.show()

""" __update_x()
update x by adding vdt on state, unless the fixed flag is present.
DO NOT call this from the outside, as it only partially updates the state.
input: state s, dt
output: new state s2
"""
def __update_x(s: list, dt: float) -> list:
    s2 = deepcopy(s)
    for o in s2:
        if o.fixed:
            continue
        o.x = o.x + o.v * dt
    return s2


""" __update_v()
update v by adding f/m * dt on state, unless the fixed flag is present.
DO NOT call this from the outside, as it only partially updates the state.
input: force array fl, state s, dt
output: new state s2
"""
def __update_v(s: list, fl: list, dt: float) -> list:
    s2 = deepcopy(s)
    num = len(s2)
    for i in range(num):
        if s2[i].fixed:
            continue
        f = fl[i]
        a = f / s2[i].m # acceleration.
        s2[i].v = s2[i].v + a * dt
    return s2


""" te_euler()
time evolution function using the Euler method.
input: original state s, t, dt
output: new time t, updated state s2
"""
def te_euler(s: list, t: float, dt: float) -> tuple[float, list]:
    num = len(s)
    s2 = deepcopy(s) # No funny business with references
    fl = [gravity(s, i) for i in range(num)] # array of force vectors.
    s2 = __update_x(s2, dt) 
    s2 = __update_v(s2, fl, dt)
    t = t + dt
    return t, s2


""" te_leapfrog()
time evolution function using the leapfrog method.
input: original state s, t, dt
output: new time t, updated state s2
"""
def te_leapfrog(s: list, t: float, dt: float) -> tuple[float, list]:
    s2 = deepcopy(s)
    num = len(s2)
    s2 = __update_x(s2, dt / 2)
    t = t + dt / 2
    fl = [gravity(s2, i) for i in range(num)]
    s2 = __update_v(s2, fl, dt)
    s2 = __update_x(s2, dt / 2)
    t = t + dt / 2
    return t, s2


# Test function where I do all the dirty stuff.
def test():
    state = []
    st = []
    hist = []
    state = add_obj(state, x = [0, 0, 0], v = [0, -0.001, 0], m = 10000, fixed = True)
    state = add_obj(state, x = [100, 0, 0], v = [5, 8, 3], m = 1)
    t = 0
    dt = 0.02
    T = 1000
    N = int(T / dt)
    for it in range(N):
        hist.append(state)
        st.append(t)
        t, state = te_leapfrog(state, t, dt)
    plot_helper(hist, st, n = 1)

# entry point
def main():
    if not (D == 2 or D == 3):
        raise Exception(f"Dimension error: D = {D}. Only 2D/3D works.")
    test()

if __name__ == "__main__":
    main()
