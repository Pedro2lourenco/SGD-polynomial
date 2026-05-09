import numpy as np
import matplotlib.pyplot as plt


plt.rcParams.update({
    "font.family": "serif",
    "font.serif":['Times New Roman'],
    "font.size": 12,
    "axes.linewidth": 1.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "legend.frameon": True
})

s0 = 123456789
s1 = 987654321

def rotl(x, k):
    return ((x << k) & 0xFFFFFFFFFFFFFFFF) | (x >> (64 - k))

def next_xorshiro():
    global s0, s1
    
    result = (s0 + s1) & 0xFFFFFFFFFFFFFFFF
    
    s1 ^= s0
    s0 = (rotl(s0, 55) ^ s1 ^ (s1 << 14)) & 0xFFFFFFFFFFFFFFFF
    s1 = rotl(s1, 36)
    
    return result

def rand_uniform():
    return (next_xorshiro() >> 11) / (1 << 53)

x = np.linspace(-3,3,100)
y = 3*x**3 - 2*x**2 + x + 3

y += rand_uniform()

a,b,c,d = rand_uniform(),rand_uniform(),rand_uniform(),rand_uniform()

lr = 5e-4
epchs = 20

for epoch in range(epchs):

    idx = np.random.permutation(len(x))

    for step, i in enumerate(idx):

        xi = x[i]
        yi = y[i]

        yhat = a*x**3 + b*x**2 + c*x + d

        err = y - yhat

        da = -2 * xi**3 * err
        db = -2 * xi**2 * err
        dc = -2 * xi * err
        dd = -2 * err

        a -= lr*da
        b -= lr*db
        c -= lr*dc
        d -= lr*dd

        plt.clf()

        plt.scatter(x, y, s=15,label='Data')

        y_pred = a*x**3 + b*x**2 + c*x + d

        plt.plot(x, y_pred,'--',color='r',label='Predict line')

        plt.xlabel('x')
        plt.ylabel('y')

        plt.xlim(min(x), max(x))
        plt.ylim(min(y)-5, max(y)+5)

        plt.pause(0.01)


plt.show()