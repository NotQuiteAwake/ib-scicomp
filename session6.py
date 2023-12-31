# session6.py
# Simulation of collision in 1D

"""
Plan
I initially wanted to do 2D, but seeing how nasty the equations are, I gave up.
The events below refers to collisions.

Adopt RHS == + convention.

Provided the implementation is correct no two particles may exchange positions. Therefore we may optimise by sorting the particles.

Data structure:
    st: list (time corresponding to events)
    hist: list (a list of all states at events)
        | state: list (list of all particles at an instant)
            | class obj (particle class)
                | x: float (position)
                | p: float (momentum)
                | im: float (inverse mass)
                | r: float = 0.05 (radius, in reality half width.)
                __init__(self, x, p, im, r = 0.05)

    event: class (info about next collision)
        | t: float (time)
        | pairs: list (list of next pairs of particles to collide.)

Functions:
    get_next(s: list) -> event:
        | Takes a state and find the next collision event.
        | Returns a event object.
    
    _collide(s: list, c: event) -> list:
        | Collide particles specified in c with each other, ASSUMING they indeed overlap. 
        | Returns new state.

    evol(s: list, dt) -> list:
        | Evolve state by dt assuming NO COLLISION.
        | Returns the new state.

    evol_col(s: list, c: event) -> list:
        | Evolve state in time, then collide particles.
        | Return new state.
    
    interpolate(hist: list, st: list) -> list:
        | Takes a hist of events to produce a history of FIXED INTERVALS in time.

    plot_x(s: list, **kwargs) -> ???:
        | Takes a state and returns a graph???
    
    animate(s: list, f: list, r: tuple):
        | Animate collision.
        | r: tuple is the range for the plot.

    main():
        | Entry point
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numpy import array, dot, linspace
from copy import deepcopy
from math import sqrt
import numpy as np

INF = 1e9
EPS = 1e-5 # A small quantity, represents error threshold.
N = int(1e8) # max number of iterations

class obj:
    x: float
    v: float
    p: float
    E: float
    im: float
    r: float = 0
    cloak: bool = False


    # Reflect changes in x / p in E / v. ASSUMES im have been initialised.
    def update(self, x: float = None, p: float = None):
        self.x = x if not x == None else self.x
        self.p = p if not p == None else self.p
        self.v = self.p * self.im
        self.E = self.p ** 2 * self.im / 2

    def __init__(self, x: float, p: float, im: float, r: float = 0, cloak: bool = False):
        self.im = im
        self.r = r
        self.update(x, p)
        self.cloak = cloak


class event:
    dt: float = INF
    pairs: list = []
    
    def __init__(self, dt: float = INF, pairs: list = []):
        self.dt = dt
        self.pairs = pairs
    
    def update(self, dt: float, p: tuple):
        # new event happens earlier
        if (dt < self.dt):
            self.dt = dt
            self.pairs = [p]
        
        # happens at the same time - collide in the same event.
        if (dt == self.dt):
            self.pairs.append(p)


def get_next(s: list) -> event:
    e = event() # container to record events

    for i in range(len(s) - 1):
        # Take out two adjacent objects
        o = s[i]
        o2 = s[i + 1]
        dv = o2.v - o.v # speed diff
        dx = (o2.x - o2.r) - (o.x + o.r) # displacement: Remember the radius.
        
        # Objects may be "misplaced" by tiny amounts after the last collision as they were evolved in time separately and then collided so their positions carry different errors. This is fine. As they have just collided and are heading away there's no need to worry.
        if (-EPS < dx < 0 and dv > 0):
            continue
        # List should be ordered!
        elif (dx < -EPS):
            raise Exception(f"""OOF!
This could mean:
Either the initial condition contains overlapping objects; 
Or my algorithm's doing something stupid.
dx = {dx}
dv = {dv}
                            """)

        # Moving apart, ignore.
        if (o2.v - o.v >= 0):
            continue
        
        dt = - dx / dv # v < 0
        e.update(dt, (i, i + 1))

    return e


def _collide(s: list, e: event) -> list:
    s2 = deepcopy(s)
    # p for pair.
    for p in (e.pairs):
        # Take out actual elements from index in the pair.
        ol = [s[x] for x in p]
        o2l = [s2[x] for x in p]
        # same equations, exchange labels with black magic
        for i in range(2):
            j = (i + 1) % 2
            # Take out the variables
            ni = ol[i].im
            pi = ol[i].p
            nj = ol[j].im
            pj = ol[j].p
            n = ni + nj

            np = (nj-ni)/n*pi + 2*nj/n*pj
            # Update momentum.
            o2l[i].update(p = np)
    return s2


def evol(s: list, t: float, dt: float) -> tuple[list, float]:
    s2 = deepcopy(s)
    for o in s2:
        o.update(x = o.x + o.v * dt)
    return s2, (t + dt)


def evol_col(s: list, t: float, e: event) -> tuple[list, float]:
    s2 = deepcopy(s)
    s2, t = evol(s2, t, e.dt)
    s2 = _collide(s2, e)
    return s2, t


# get total energy, not to be confused with e for event.
def get_E(s: list) -> float:
    E = 0
    for o in s:
        E = E + o.E
    return E


# get total momentum of a state.
def get_p(s: list) -> float:
    p = 0
    for o in s:
        p = p + o.p
    return p


def interpolate(hist: list, st: list, dt:float, T: float) -> tuple[list, list]:
    f = [] # frames, a list of states of equal spacing in time.
    ft = [] # time at each frame
    t = 0 # iteration variable
    N = int(T / dt) # steps of iteration
    pt = 1 # pointer to next collision
    s = hist[0] # initial state
    f.append(s)
    ft.append(t)

    # main loop
    for i in range(N):
        delta = dt # actual evolution interval
        # Have just passed the time for another collision.
        while (pt < len(st) and st[pt] <= t + delta):
            # Look ahead and move past the collisions that took place before our next iteration.
            s = hist[pt]
            delta = t + delta - st[pt] # evolution interval is shortened here.
            t = st[pt] # So time can be moved forward
            pt = pt + 1
        
        s, t = evol(s, t, delta)
        f.append(s)
        ft.append(t)

    return f, ft


# animating function for use in FuncAnimation in gen_anim
# r is the range for the axis.
def animate(i: int, fig, f: list, ft: list, r: tuple[float, float]):
    fig.clf()
    ax = fig.add_subplot()
    ax.set_xlim(r[0], r[1])
    ax.set_ylim(-1, 1)
    ax.set_title(f"t = {round(ft[i], 2)}") # Time is rounded to 2dp

    for o in f[i]:
        if o.cloak:
            continue
        x = o.x
        ax.plot(x, 0, 'o')


def gen_anim(f: list, ft: list, L: float, R: float, dt: float): 
    # Animate
    anim_fig = plt.figure()

    # interval reflects real time
    anim = animation.FuncAnimation(anim_fig, animate, fargs = (anim_fig, f, ft, (L, R)), frames = len(f), interval = 1000 * dt)
    # In reality real time seems to only work with fps settings.
    writevid = animation.FFMpegWriter(fps = int(1/dt))

    anim.save("col.mp4", writer=writevid, dpi=240, progress_callback = lambda i, n: print(f'Animation: {round(i/n*100, 2)}%'))


def plot_xt(hist: list, st: list, L: float, R: float, T: float):
    for i in range(len(hist[0])):
        if hist[0][i].cloak:
            continue
        plt.plot(st, [s[i].x for s in hist])
    
    plt.title("x - t graph of all particles")    
    plt.xlim(0, T)
    plt.ylim(L, R)
    plt.show() 


# T: total time
def sim(state: list, T: float, dt: float) -> tuple[list, list]:
    # algo relies on states being sorted for O(n)
    state = sorted(state, key=(lambda o : o.x))
    t = 0 # current time in simulation
    st = [] # list of time
    hist = [] # list of event states

    # N defines maximum number of simulation steps.
    for i in range(N):
        hist.append(state)
        st.append(t)

        # print(get_E(state)) # energy check

        # simulation finished
        if (t > T):
            break
        e = get_next(state)
        # no more collisions
        if (len(e.pairs) == 0):
            break

        state, t = evol_col(state, t, e)
    
    return st, hist

"""
Physics!
Below are an assortment of functions that investigates the physics. At the end of the file is the main() function.
"""
# Takes a list of states and the corresponding times (either (f, ft) or (hist, st)) and check energy conservation across them.
def E_conservation(f: list, ft: list) -> bool:
    E = get_E(f[0])
    for s, t in zip(f, ft):
        if abs(E - get_E(s)) > EPS:
            print(f"Energy conservation fails at {t}")
            return False

    print("Energy conservation, check.")
    return True


# Takes a list of states and check momentum conservation. Almost incredibly this still works for situations with walls thanks to use of inverse masses (I thought it would fail and debugged for quite some time :(((  ).
def p_conservation(f: list, ft: list) -> bool:
    p = get_p(f[0])
    for s, t in zip(f, ft):
        if abs(p - get_p(s)) > EPS:
            print(f"Momentum conservation fails at {t}")
            return False

    print("Momentum conservation, check.")
    return True

def histogram_plot():
    state = []
    
    # Location of walls
    L = -100
    R = 100
    T = 1000
    dt = 0.1
    
    # radius for graph to look nice
    r = 0.175

    # Initial conditions
    state.append(obj(-90, 5, 1, r = r)) # 1
    state.append(obj(-60, -3, 2, r = r))
    state.append(obj(-30, 100, 0.01, r = r))
    state.append(obj(10, 2, 4, r = r)) # 4
    state.append(obj(20, -20, 1, r = r)) # 5
    state.append(obj(50, 1, 10, r = r))
    state.append(obj(60, -1, 0.1, r = r))
    state.append(obj(90, 0.01, 1000, r = r)) # 8

    # Add in walls
    state.append(obj(L, 0, 0, r = 0, cloak = True))
    state.append(obj(R, 0, 0, r = 0, cloak = True))

    st, hist = sim(state, T, dt)

    plot_xt(hist, st, L, R, T)

    f, ft = interpolate(hist, st, dt, T) 

    """
     Velocity distributions: It was observed that the velocity spread of the 1/4 particle is about twice that of the m = 1 particle. 
     This may be understood with the equipartition theorem, if we interpret this spread as a measure of mean speed. Then we have the 1/4 particle has 4 times v^2 on average, and therefore the same average energy as the m = 1 particle.
    """
    
    v4 = [s[4].v for s in hist]
    v5 = [s[5].v for s in hist]
 
    bins = linspace(-50, 50, 50)
    plt.hist(v4, bins, alpha = 0.5, label='$m = 1/4$')
    plt.hist(v5, bins, alpha = 0.5, label='$m = 1$')
    plt.legend(loc='upper right')
    plt.title("velocity distributions")
    plt.show()

    # Position of the very light #8. The distribution is heavily skewed away from the very heavy #7 particle and close to the wall.

    x8 = [s[8].x for s in hist]
    bins = linspace(25, R, 100)
    plt.hist(x8, bins, label = 'm = 0.001')
    plt.title("Position distribution of ligth particle m = 0.001")
    plt.show()


def moving_wall():
    state = []
    
    # Location of walls
    L = -100
    R = 100
    T = 1000
    dt = 0.1
    
    # radius for animation to look nice
    r = 0 # No animation here.

    # Initial conditions
    state.append(obj(-90, 5, 1, r = r)) # 1
    state.append(obj(-60, -3, 2, r = r))
    state.append(obj(-30, 100, 0.01, r = r))
    state.append(obj(10, 2, 4, r = r)) # 4
    state.append(obj(20, -20, 1, r = r)) # 5
    state.append(obj(50, 1, 10, r = r))
    state.append(obj(60, -1, 0.1, r = r))
    state.append(obj(90, 0.01, 1, r = r)) # 8

    # Moving wall / #9: It will be annoting to implement actual infinite mass moving wall (a lot of special treatments that would otherwise be useless, ZMF, etc.) but this should approximate the correct behaviour.
    state.append(obj(L, 0, 0, r = 0, cloak = True))
    state.append(obj(R, INF / 10, 1/INF, r = 0, cloak = False)) #9

    st, hist = sim(state, T, dt)

    # Right wall at v = 0.1, moves 100 in 1000s.
    plot_xt(hist, st, L, R + 100, T)

    f, ft = interpolate(hist, st, dt, T) 

    Elist = array([get_E(s[:-1]) for s in f]) # Don't get the RHS wall involved, since we want energy that remains in the system.

    W = array([max(Elist)]*len(Elist)) - Elist
    x = array([s[9].x for s in f])

    plt.plot(ft, Elist)
    plt.title("Energy of system against time")
    plt.show()
    plt.clf()

    plt.plot(x ** -2, W)
    plt.title("Work done by system against x^-2.")
    plt.show()

    """ Analysis
    As the wall is stretched out over 1000s through 100m, the energy dropped from 284 to 124. From this the gamma value can be deduced. From wikipedia gamma = 1 + 2/f, where f is the degrees of freedom so here gamma = 3.
    It can then be shown that work done by gas against displacement ^ (1 - gamma) should give a straight line: A plot of W against x^-2 seems to support gamma = 3
    """


def animation_demo():
    state = []
    
    # Location of walls
    L = -10
    R = 10
    T = 20
    dt = 0.01
    
    # radius for animation to look nice (with some tinkering...)
    r = 0.175

    # Initial conditions
    state.append(obj(-5, 5, 1, r = r))
    state.append(obj(0, -0.5, 20, r = r))
    state.append(obj(5, 4, 0.5, r = r))
    
    # Add in walls
    state.append(obj(L, 0, 0, r = 0, cloak = True))
    state.append(obj(R, 0, 0, r = 0, cloak = True))

    st, hist = sim(state, T, dt)

    print("An animation called col.mp4 will be saved to the local directory. If you don't fancy that, Ctrl-c before closing the graph. ")
    plot_xt(hist, st, L, R, T)

    f, ft = interpolate(hist, st, dt, T)
    gen_anim(f, ft, L, R, dt)
    
def main():
    print("I have written three examples; Comment / uncomment them in main() to see them in action.")
    # Uncomment one of these functions to see some demonstrations of the code.
    histogram_plot()
    # moving_wall()
    # animation_demo() # saves an animation called col.mp4 into the same folder. 

    return

if __name__ == "__main__":
    main()
