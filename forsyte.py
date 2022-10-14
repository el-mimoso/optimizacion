from distutils.log import debug
import matplotlib.pyplot as plt
import numpy as np
import math
from timeit import default_timer as timer


# función original f(x) = x_1 e^(-x_1^2 - x_2^2)
# xini  [-1,-1]

def f(x):
    f = 0
    y = [1, 2, 3, 4, 5, 4, 3, 2, 1, 0]
    for i in range(10):
        aux = x[i] - y[i]
        f += aux**2
        if i > 0:
            aux = x[i]-x[i-1]
            f += 2.5*aux**2
    return f


# gradiente  ∇f
def grad(x):
    # list1 = [1-2*x[0]**2, -2*x[0]*x[1]]
    # descenso = np.array(list1)*math.e**(-x[0]**2 - x[1]**2)
    # return descenso
    y = [1, 2, 3, 4, 5, 4, 3, 2, 1, 0]
    g = np.array([None]*10)
    for i in range(10):
        g[i] = 2.0*(x[i]-y[i])
        if i > 0:
            g[i] += 5*(x[i]-x[i-1])
        if i < 10-1:
            g[i] += 5*(x[i]-x[i+1])
    return g


# gradiente  ∇^2f
def hessiano(x):
    # return axay
    return np.array([
        [7, -5, 0, 0, 0, 0, 0, 0, 0, 0],
        [-5, 12, -5, 0, 0, 0, 0, 0, 0, 0],
        [0, -5, 12, -5, 0, 0, 0, 0, 0, 0],
        [0, 0, -5, 12, -5, 0, 0, 0, 0, 0],
        [0, 0, 0, -5, 12, -5, 0, 0, 0, 0],
        [0, 0, 0, 0, -5, 12, -5, 0, 0, 0],
        [0, 0, 0, 0, 0, -5, 12, -5, 0, 0],
        [0, 0, 0, 0, 0, 0, -5, 12, -5, 0],
        [0, 0, 0, 0, 0, 0, 0, -5, 12, -5],
        [0, 0, 0, 0, 0, 0, 0, 0, -5, 7]
    ])


# dirección del gradiente p
def dirgrad(x):
    vgrad = grad(x)
    magGrad = np.sqrt(vgrad.dot(vgrad))
    p = -vgrad/magGrad
    return p


def phiAlpha(x0, alpha, p):
    paX = x0 + p * alpha
    return f(paX)


def phipAlpha(x0, alpha, p):
    x = x0 + alpha * p
    vgrad = grad(x)
    return(np.dot(vgrad, p))


def phipp(x0, alpha, p):
    x = x0 + alpha * p
    ahess = hessiano(x)
    return np.dot(np.dot(ahess, p), p)


def exhaustivoRefinado(p, xini, alpha=0, h=0.1, tol=1e-6):
    """Busqueda de minimo con metodo exhaustivo refinado. puedes cambiar el paso
    Retorna f(a) y alpha
    """
    k = 0
    while h > tol:
        while phiAlpha(xini, alpha+h, p) < phiAlpha(xini, alpha, p):
            alpha = alpha + h
            fnow = phiAlpha(xini, alpha, p)
            # print(k, h, fnow)
            k += 1
        alpha = alpha-h
        h = h / 10
    return alpha


def gradDescent(x0):
    p = dirgrad(x0)
    alpha = exhaustivoRefinado(p, x0)
    # TODO: buscar alpha con newton para mayor precisión ?
    x0 = x0 + alpha*p
    return x0


def forsyte(x0, k=0, m=0, tol=1e-4):
    """Algoritmo de forsyte."""
    print("k, x^(k), p^(k), f(x^k), t")
    while np.linalg.norm(grad(x0)) >= tol:
        x1 = gradDescent(x0)
        x2 = gradDescent(x1)
        y = x2
        d = (y - x0)/np.linalg.norm(y - x0)
        alpha = exhaustivoRefinado(d, x0)
        # TODO: buscar alpha con newton para mayor precisión ?
        # print(f"alpha: {alpha}")
        x0 = x0 + alpha*d
        itTime = timer()
        print(f"{k}, {x0}, {d} , {f(x0)},{itTime}")
        k = k + 1
    return x0


x0 = [1, 2, 3, 4, 5, 4, 3, 2, 1, 0]

start = timer()
print(start)
xfin = forsyte(x0)
end = timer()
print(end)
print(f"Tiempo de ejecución: {end-start} s")
print(f"Evaluacion del mínimo: {grad(xfin)}")
