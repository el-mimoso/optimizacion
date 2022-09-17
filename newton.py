import matplotlib.pyplot as plt
import numpy as np
import math


tol = 1e-6


def fx(x):
    return -2*math.sin(x) + x**2/10


def fp(x):
    res = -2*math.cos(x) + x/5
    return res


def fpp(x):
    return 2*math.sin(x) + 1/5


def st(a, x):
    return fx(a)+(fp(a)*(x-a))+((fpp(a)*(x-a)**2)/2)


st = np.vectorize(st)


def newton(xo, itmax):
    k = 0
    while abs(fp(xo)) > tol or k > itmax:
        fprima = fp(xo)
        fbiprima = fpp(xo)
        xk = xo - (fprima/fbiprima)
        print(
            f"k: {k}  , xk: {round(xk, 6)} , fx:{round(fx(xk), 6)}  , fp: {round(fprima,6)} , fpp: {round(fbiprima,6)}")
        k = k+1
        xo = xk


newton(0.5, 20)


x = np.linspace(-2, 4, 50)

plt.subplot(2, 2, 1)
plt.plot(x, -2*np.sin(x)+(x**2/10), label="f(x)")
plt.title('Función')
plt.legend(loc=2)
plt.grid()


plt.subplot(2, 2, 2)
plt.plot(x, -2*np.sin(x)+(x**2/10), label="f(x)")
plt.plot(x, st(0.5, x), 'm--', label="q(x)")
plt.title('Primera Iteración')
plt.legend(loc=2)

plt.grid()

plt.subplot(2, 2, 3)
plt.plot(x, -2*np.sin(x)+(x**2/10), label="f(x)")
plt.plot(x, st(1.928281, x), 'g--', label="q(x)")
plt.title('Segunda Iteración')
plt.legend(loc=2)

plt.grid()

plt.subplot(2, 2, 4)
plt.plot(x, -2*np.sin(x)+(x**2/10), label="f(x)")
plt.plot(x, st(1.404788, x), 'r--', label="q(x)")
plt.title('Tercera Iteración')
plt.legend(loc=2)

plt.grid()

plt.tight_layout()

plt.show()
