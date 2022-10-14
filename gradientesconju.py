import matplotlib.pyplot as plt
import numpy as np
import math
from timeit import default_timer as timer


# función original f(x) = x_1 e^(-x_1^2 - x_2^2)
# xini  [-1,-1]
def f(x1, x2):
    a = 4*x1**2 + x2**2
    return a


# gradiente  ∇f
def grad(x1, x2):
    list1 = [4 * x1 + 2*x2 +3, 2*x1 +20*x2 -4]
    descenso = np.array(list1)
    return descenso


# dirección del gradiente p
def dirgrad(x1, x2):
    vgrad = grad(x1, x2)
    magGrad = np.sqrt(vgrad.dot(vgrad))
    p = -vgrad/magGrad
    return p


def phiAlpha(x0, alpha, p):
    paX1 = x0[0] + p[0] * alpha
    paX2 = x0[1] + p[1] * alpha
    return f(paX1, paX2)


def phipAlpha(x0, alpha, p):
    x1 = x0[0] + alpha * p[0]
    x2 = x0[0] + alpha * p[1]
    vgrad = grad(x1, x2)
    return(np.dot(vgrad, p))


def phipp(x0, alpha, p):
    val = []
    for a in alpha:
        x1 = x0[0] + a * p[0]
        x2 = x0[0] + a * p[1]
        ahess = hessiano(x1, x2)
        val.append(np.dot(np.dot(ahess, p), p))
    return val


def hessiano(x, y):
    # axax = (2*x*((2*x**2)-3))*math.e**(-(x**2) - (y**2))
    # ayay = (2*x*((2*y**2) - 1))*math.e**(-(x**2) - (y**2))
    # axay = 2*(2*x**2 - 1)*y*math.e**(-x**2 - y**2)
    # return axay
    return np.array([
        [4, 2],
        [2, 20]
    ])



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

# x = np.linspace(0, 2, 50)
x0List = [20, 30]
x0 = np.array(x0List)
r = grad(x0[0], x0[1])
p = r*-1

def almd (r, p):
    return -np.dot(r, p) / np.dot(np.dot(hessiano(x0List[0], x0List[1]), p), p)

print(f"r:{r} p:{p}")

aMD = almd(r,p)
print(f"paso 1 AlphaMD: {aMD}")

def x(x0, aMD, p ):
    return x0 + aMD*p

xk1= x(x0List,aMD,p)
print(f"paso 2 x {xk1}")

def beta(r,p):
    return np.dot(np.dot(hessiano(x0List[0], x0List[1]), p), r)/np.dot(np.dot(hessiano(x0List[0], x0List[1]), p), p)

b = beta(r,p)
print(f"paso 3 Beta: {b}")

pn = -r + b*p
print(f"paso 4 Calcular p: {pn}")

aMD = almd(p*-1,p)
print(x(xk1,aMD,p))


# -0.8947368421, 0.28947368421







# iniciamos cronometro en 0's
# start = timer()
# xfin = forsyte(x0)
# end = timer()
# print(f"Tiempo de ejecución: {end-start} s")
# print(f"Evaluacion del mínimo: {grad(xfin[0], xfin[1])}")
# print(f(xfin[0], xfin[1]))
# gradDescent(x0)
# p = dirgrad(x0[0], x0[1])
