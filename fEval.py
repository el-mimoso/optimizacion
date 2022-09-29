import numpy as np
import math


# función original f(x) = x_1 e^(-x_1^2 - x_2^2)


def f(x1, x2):
    #Funición a evaluar
    a = (np.e**(-x1**2 - x2**2))*x1
    return a


def grad(x1, x2):
    """ Retorna gradiente  ∇f"""
    list1 = [1-2*x1**2, -2*x1*x2]
    descenso = np.array(list1)*math.e**(-x1**2 - x2**2)
    return descenso


def dirgrad(x1, x2):
    """Retorta la dirección del gradiente p"""
    vgrad = grad(x1, x2)
    magGrad = np.sqrt(vgrad.dot(vgrad))
    p = -vgrad/magGrad
    return p


def hessiano(x, y):
    """Evalua la matriz Hessiana, ingresar derivadas"""
    axax = (2*x*((2*x**2)-3))*math.e**(-(x**2) - (y**2))
    ayay = (2*x*((2*y**2) - 1))*math.e**(-(x**2) - (y**2))
    axay = 2*(2*x**2 - 1)*y*math.e**(-x**2 - y**2)
    # return axay
    return np.array([
        [axax, axay],
        [axay, ayay]
    ])


# valores de x inicial y paso
xini = [-1, -1]

# calculamos el grad y su dirección
g = grad(xini[0], xini[1])
p = dirgrad(xini[0], xini[1])
alpha = 0
k = 0


def exhaustivo(p, xini, alpha, k, h=0.05,):
    fantes = f(xini[0], xini[1])
    while True:
        alpha = alpha + h
        x = xini + alpha*p
        fnow = f(x[0], x[1])
        print(k, x[0], x[1], fnow)
        k = k+1
        if (fnow > fantes):
            break
        fantes = fnow


def maximoDescenso(xini):
    """Regresa alpha maximo descenso """
    vgrad = grad(xini[0], xini[1])
    amaxDes = -np.dot(vgrad, p) / \
        np.dot(np.dot(hessiano(xini[0], xini[1]), p), p)
    return amaxDes


def phiAlpha(x0, x, d):
    paX1 = x0[0] + x * d[0]
    paX2 = x0[1] + x * d[1]
    return f(paX1, paX2)


def phipAlpha(x0, alpha,  p):
    x1 = x0[0] + alpha * p[0]
    x2 = x0[0] + alpha * p[1]
    vgrad = grad(x1, x2)
    return(np.dot(vgrad, p))

print("gradiente")
print(g)
print("dir p")
print(p)
print(maximoDescenso(xini))
