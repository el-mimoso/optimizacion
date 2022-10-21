import matplotlib.pyplot as plt
import numpy as np
import math
from timeit import default_timer as timer


#Función, gradiente y Hessiano
def f(x):
    """Funcion a Evaluar"""
    f = 2*x[0]**2+2*x[0]*x[1]+10*x[1]**2 + 20 + 3*x[0]-4*x[1]
    return f


# gradiente  ∇f
def grad(x):
    g = np.array([
        4*x[0] + 2*x[1] + 3,
        2*x[0] + 20*x[1] - 4
    ])
    return g


def hessiano(x):
    # return axay
    return np.array([
        [4, 2],
        [2, 20]
    ])


# dirección del gradiente p
def dirgrad(x):
    vgrad = grad(x)
    magGrad = np.sqrt(vgrad.dot(vgrad))
    p = -vgrad/magGrad
    return p

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
    return (np.dot(vgrad, p))


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


def almd(x0, r, p):
    return -np.dot(r, p) / np.dot(np.dot(hessiano(x0), p), p)

def beta(x0, r, p):
    return np.dot(np.dot(hessiano(x0), p), r)/np.dot(np.dot(hessiano(x0), p), p)


def gradConjugadoPreliminar(x0, b, k=0, tol=1e-6):
    r = grad(x0)
    p = r*-1
    print(x0)
    print("x0, f(x^k), aMD, b")
    while np.linalg.norm(grad(x0)) >= tol:
        aMD = almd(x0, r, p)
        x0 = x0 + aMD*p
        r = np.dot(hessiano(x0), x0) - b
        b = beta(x0, r, p)
        p = -r + b*p
        print(x0, f(x0), aMD, b)
    return x0


def gradienteConjugado(x0, b, k=0, tol=1e-6):
    r = grad(x0)
    p = r*-1
    print(x0)
    print("x0, f(x^k), aMD, b")
    while np.linalg.norm(grad(x0)) >= tol:
        alpha = np.dot(r, r) / np.dot(np.dot(hessiano(x0), p), p)
        x0 = x0 + alpha*p
        r1 = r + alpha * np.dot(hessiano(x0), p)
        b = (np.dot(r1,r1))/(np.dot(r,r))
        p = -r1 + b*p
        print(x0, f(x0), alpha, b)
        r = r1
    return x0


x0 = np.array([20, 30])
b = [-3, 4]
print(gradConjugadoPreliminar(x0, b))
print(gradienteConjugado(x0,b))


# -0.8947368421, 0.28947368421

