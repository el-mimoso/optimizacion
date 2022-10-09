import matplotlib.pyplot as plt
import numpy as np
import math


# función original f(x) = x_1 e^(-x_1^2 - x_2^2)
# xini  [-1,-1]

def f(x1, x2):
    a = (np.e**(-x1**2 - x2**2))*x1
    return a


# gradiente  ∇f
def grad(x1, x2):
    list1 = [1-2*x1**2, -2*x1*x2]
    descenso = np.array(list1)*math.e**(-x1**2 - x2**2)
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
    axax = (2*x*((2*x**2)-3))*math.e**(-(x**2) - (y**2))
    ayay = (2*x*((2*y**2) - 1))*math.e**(-(x**2) - (y**2))
    axay = 2*(2*x**2 - 1)*y*math.e**(-x**2 - y**2)
    # return axay
    return np.array([
        [axax, axay],
        [axay, ayay]
    ])


f = np.vectorize(f)


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
    print("k, x^(k), p^(k), f(x^(k), θ")
    op = dirgrad(x0[0], x0[1])
    while np.linalg.norm(grad(x0[0], x0[1])) >= tol:
        p = dirgrad(x0[0], x0[1])
        alpha = exhaustivoRefinado(p, x0)
        # print(f"a: {alpha}")
        x0 = x0 + alpha*p
        if k >= 1:
            angulo = np.arccos(np.dot(op, p))
            op = p
            print(f"{k}, {x0}, {p} , {f(x0[0],x0[1])}, {round(np.degrees(angulo),6)} ")
        else:
            print(f"{k}, {x0}, {p} , {f(x0[0],x0[1])}, - ")

        k = k+1


x = np.linspace(0, 2, 50)
x0List = [-1, -1]
x0 = np.array(x0List)
gradDescent(x0)
p = dirgrad(x0[0], x0[1])

rwlf = []
pphi = []

for alpha in x:
    rwlf.append(rectaWlf(x0, alpha, p, 1e-4))
    cwlf = cw(x0, p*-1, alpha)
    pphi.append(phipAlpha(x0, alpha, p))
    # print(
    # f"alpha:{round(alpha,4)} curva: {cwlf[0]} armijo: {cwlf[1]}  wolfe: {cwlf[2]}")


# plt.plot(x, phiAlpha(x0, x, p), 'b', label="phi(alpha)")
# # plt.plot(x, pphi, 'r--', label="phi(alpha)")
# plt.plot(x, rwlf, 'b--', label="phi(alpha)")
# plt.grid()
# plt.show()
