from distutils.log import debug
import matplotlib.pyplot as plt
import numpy as np
import math
from timeit import default_timer as timer


# función original f(x) = x_1 e^(-x_1^2 - x_2^2)
# xini  [-1,-1]

def f(x):
    f = x[0]**2+2*x[1]**2-math.cos(3*math.pi*x[0])-math.cos(4*math.pi*x[1])+0.7
    return f


# gradiente  ∇f
def grad(x):
    g = np.array([2*x[0] + 9.42477796076938*math.sin(9.42477796076938*x[0]),
                  4*x[1] + 12.5663706143592*math.sin(12.5663706143592*x[1])])
    return g


def hessiano(x):
    return np.array([
        [88.8264396098042*math.cos(9.42477796076938*x[0]) + 2, 0],
        [0, 157.91367041743*math.cos(12.5663706143592*x[1]) + 4]
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


def exhaustivoRefinado(p, xini, alpha=0, h=0.1, tol=1e-9):
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


def forsyte(x0, k=0, m=0, tol=1e-2):
    """Algoritmo de forsyte."""
    print("k, x ^ (k), f(x ^ (k), pk")
    while np.linalg.norm(grad(x0)) >= tol:
        x1 = gradDescent(x0)
        x2 = gradDescent(x1)
        x3 = gradDescent(x2)
        y = x3
        d = (y - x0)/np.linalg.norm(y - x0)
        alpha = exhaustivoRefinado(d, x0)
        # TODO: buscar alpha con newton para mayor precisión ?
        # print(f"alpha: {alpha}")
        x0 = x0 + alpha*d
        itTime = timer()
        print(f"{k}, {x0}, {f(x0)}, {d} ")
        k = k + 1
    return x0


x0 = [2, 3]
print("Forsyte")
start = timer()
print(start)
xfin = forsyte(x0)
end = timer()
print(end)
print(f"Tiempo de ejecución: {end-start} s")
print(f"Evaluacion del mínimo: {grad(xfin)}")
