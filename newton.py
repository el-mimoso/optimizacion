import matplotlib.pyplot as plt
import numpy as np
import math


tol = 1e-3


def ecuacion(x):
    return -2*math.sin(x) + x**2/10


def fp(x):
    return -2*math.cos(x) + x/5


def fpp(x):
    return 2*math.sin(x) + 1/5


def newton(xo, itmax):
    k = 0
    while k <= itmax:
        fprima = fp(xo)
        fbiprima = fpp(xo)
        xk = xo - (fprima/fbiprima)
        print(f"k: {k}  , xk: {round(xk, 6)}, fp: {fprima}, fpp: {fprima}")
        k = k+1
        xo = xk


newton(0, 20)


x = np.linspace(-5, 5, 50)
plt.plot(x, -2*np.sin(x)-(x**2/10), label="f(x)")
plt.plot(x, -2*x+(x ** 2)/10, 'm', label="q(x)")
plt.legend(loc=2)
plt.grid()
plt.show()
