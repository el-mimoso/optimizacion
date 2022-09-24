from cProfile import label
import matplotlib.pyplot as plt
import numpy as np
import math


# función original f(x) = x_1 e^(-x_1^2 - x_2^2)

# xini = []

def f(x1, x2):
    a = (np.e**(-x1**2 - x2**2))*x1
    return a


# phi de alpha cheat code
def fx(x):
    return (-1+x/math.sqrt(5))*math.e**(-(-1+((2*x)/math.sqrt(5)))**2 - (-1+(x/math.sqrt(5)))**2)


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

# pasar p para graficar, d para evaluar en cierto punto.
def phiAlpha(x0, x, d):
    paX1 = x0[0] + x * d[0]
    paX2 = x0[1] + x * d[1]
    print(paX1)
    print(paX2)
    # phiAl = x0 + x * d
    return f(paX1, paX2)


x0List = [-1, -1]
x0 = np.array(x0List)
p = dirgrad(x0[0], x0[1])
d = p*-1

# DEBUGG
# print("probando en phi de alfa: ")
# phiAlpha(x0, -0.952194, d)
# print(phiAlpha(x0, -0.952194, d))
#retorna -0.4039332232588827

f = np.vectorize(f)


def hessiano(x, y):
    axax = (2*x*((2*x**2)-3))*math.e**(-(x**2) - (y**2))
    ayay = (2*x*((2*y**2) - 1))*math.e**(-(x**2) - (y**2))
    axay = 2*(2*x**2 - 1)*y*math.e**(-x**2 - y**2)
    # return axay
    return np.array([
        [axax, axay],
        [axay, ayay]
    ])


# print(hessiano(1,1))

fx = np.vectorize(fx)


x = np.linspace(0, 2, 50)
# plt.plot(x, fx(x), 'g--')
plt.plot(x, phiAlpha(x0, x, p), 'b', label="phi(alpha)")
plt.grid()
plt.show()
