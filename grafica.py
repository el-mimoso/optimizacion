import matplotlib.pyplot as plt
import numpy as np
import math


# función original f(x) = x_1 e^(-x_1^2 - x_2^2)
# xini  [-1,-1]

def f(x1, x2):
    a = x1**2+2*x2**2 - \
        math.cos(3*math.pi*x1)-math.cos(4*math.pi*x2)+0.7

    return a


# phi de alpha cheat code
def fx(x):
    return (-1+x/math.sqrt(5))*math.e**(-(-1+((2*x)/math.sqrt(5)))**2 - (-1+(x/math.sqrt(5)))**2)


# gradiente  ∇f
def grad(x, y):
    list1 = [2*x + 9.42477796076938*math.sin(9.42477796076938*x),
             4*x + 12.5663706143592*math.sin(12.5663706143592*y)]
    descenso = np.array(list1)
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
    return f(paX1, paX2)


def phipAlpha(x0, x, p):
    val = []
    for alpha in x:
        x1 = x0[0] + alpha * p[0]
        x2 = x0[0] + alpha * p[1]
        vgrad = grad(x1, x2)
        val.append(np.dot(vgrad, p))
    return val


def phipp(x0, alpha, p):
    val = []
    for a in alpha:
        x1 = x0[0] + a * p[0]
        x2 = x0[0] + a * p[1]
        ahess = hessiano(x1, x2)
        val.append(np.dot(np.dot(ahess, p), p))
    return val


x0List = [2, 3]
x0 = np.array(x0List)
p = dirgrad(x0[0], x0[1])
print(p)
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


x = np.linspace(0, 3, 100)
# plt.plot(x, fx(x), 'g--')


plt.plot(x, phiAlpha(x0, x, p), 'b', label="phi(alpha)")
plt.plot(x, phipAlpha(x0, x, p), 'g', label="phi'(alpha)")
plt.plot(x, phipp(x0, x, p), 'r', label="phi''(alpha)")
plt.legend(loc=1)
plt.grid()
plt.show()
