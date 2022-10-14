import matplotlib.pyplot as plt
import numpy as np
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


# c1 1*10-4
def cw(x0, p, alpha, c1=1e-4, c2=1):

    armijo = False
    curvatura = False
    wolfe = False

    recta = phiAlpha(x0, 0, p) + c1*alpha*phipAlpha(x0, alpha, p)

    if phiAlpha(x0, alpha, p) < recta:
        armijo = True

    if np.abs(phipAlpha(x0, alpha, p)) <= c2*np.abs(phipAlpha(x0, 0, p)):
        curvatura = True

    if curvatura == True and armijo == True:
        wolfe = True
    return(curvatura, armijo, wolfe)


def rectaWlf(x0, alpha, p, c1):
    recta = phiAlpha(x0, 0, p) + c1*alpha*phipAlpha(x0, 0, p)
    return recta


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


def gradDescent(x0, k=0, tol=1e-4):
    print("k, x^(k), p^(k), f(x^(k), θ, t")
    op = dirgrad(x0)
    while np.linalg.norm(grad(x0)) >= tol:
        p = dirgrad(x0)
        alpha = exhaustivoRefinado(p, x0)
        # print(f"a: {alpha}")
        x0 = x0 + alpha*p
        if k >= 1:
            angulo = np.arccos(np.dot(op, p))
            op = p
            print(f"{k}, {x0}, {p} , {f(x0)}, {round(np.degrees(angulo),6)}, {timer()} ")
        else:
            print(f"{k}, {x0}, {p} , {f(x0)}, - , {timer()}")

        k = k+1


x = np.linspace(0, 2, 50)
x0List = [1, 2, 3, 4, 5, 4, 3, 2, 1, 0]
x0 = np.array(x0List)
start = timer()
print(start)
gradDescent(x0)
end = timer()
print(end)
print(f"Tiempo de ejecución: {end-start} s")
p = dirgrad(x0)

