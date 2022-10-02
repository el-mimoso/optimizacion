from traceback import print_tb
import matplotlib.pyplot as plt
import numpy as np
import math


def f(x1, x2):
    a = (np.e**(-x1**2 - x2**2))*x1
    return a


def grad(x1, x2):
    """ gradiente  ∇f"""
    list1 = [1-2*x1**2, -2*x1*x2]
    descenso = np.array(list1)*math.e**(-x1**2 - x2**2)
    return descenso


def dirgrad(x1, x2):
    """ dirección del gradiente p """
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


def phiAlpha(x0, x, d):
    paX1 = x0[0] + x * d[0]
    paX2 = x0[1] + x * d[1]
    return f(paX1, paX2)


def phipAlpha(x0, alpha,  p):
    x1 = x0[0] + alpha * p[0]
    x2 = x0[0] + alpha * p[1]
    vgrad = grad(x1, x2)
    return(np.dot(vgrad, p))


def phipp(x0, alpha, p):
    x1 = x0[0] + alpha * p[0]
    x2 = x0[0] + alpha * p[1]
    ahess = hessiano(x1, x2)
    return np.dot(np.dot(ahess, p), p)


def st(x0, p,a, a0=1):
    # C = phiAlpha(x0, a0, p)
    # B = phipAlpha(x0, a0, p)
    # numerador = (phiAlpha(x0, a1, p) - phipAlpha(x0, a0, p)*(a1-a0) - phiAlpha(x0, a0, p))
    # denominador = (a1-a0)**2
    # A = numerador/denominador
    # print(A)
    return phiAlpha(x0,a0,p)+(phipAlpha(x0,a0,p)*(a-a0))+((phipp(x0, a0, p)*(a-a0)**2)/2)

x0List = [-1, -1]
x0 = np.array(x0List)
p = dirgrad(x0[0], x0[1])
x = np.linspace(-2, 2, 50)
alpha = []
# st(x0,p,)

incremento = 0
# dot es la punto al rededor de la serie de taylor
dot = phiAlpha(x0, incremento, p)
print(dot)
for a in x:
    alpha.append(st(x0, p, a, incremento))

plt.plot(x, phiAlpha(x0, x, p), 'b', label="phi(alpha)")
plt.plot(x, alpha, 'r--',
         label="q(x) alpha de interpolación con alpha_0 = 0 y alpha_1 = 0.3")
plt.plot(incremento, dot, 'ro')
plt.legend(loc=1)
plt.grid()
plt.show()