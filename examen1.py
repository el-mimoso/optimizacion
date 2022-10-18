import matplotlib.pyplot as plt
import numpy as np
from timeit import default_timer as timer
import math

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


# c1 1*10-4
def cw(x0, p, alpha, c1=1e-4, c2=1):
    """Condiciones de Wolfe"""
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
    return (wolfe)


def rectaWlf(x0, alpha, p, c1):
    recta = phiAlpha(x0, 0, p) + c1*alpha*phipAlpha(x0, 0, p)
    return recta


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


def gradDescent(x0, k=0, tol=1e-6):
    print("k, x^(k), f(x^(k), pk, θ,")
    op = dirgrad(x0)
    while np.linalg.norm(grad(x0)) >= tol:
        p = dirgrad(x0)
        alpha = exhaustivoRefinado(p, x0)
        # print(f"a: {alpha}")
        x0 = x0 + alpha*p
        if k >= 1:
            angulo = np.arccos(np.dot(op, p))
            op = p
            print(f"{k}, {x0}, {f(x0)}, {p},  {round(np.degrees(angulo),6)} ")
        else:
            print(f"{k}, {x0}, {f(x0)}, {p},  - ")

        k = k+1
    return x0


def maximoDescenso(xini, p):
    """Regresa alpha maximo descenso """
    vgrad = grad(xini)
    amaxDes = -np.dot(vgrad, p) / \
        np.dot(np.dot(hessiano(xini), p), p)
    return amaxDes


def interpolacion(xo, p, a1, ao=0):
    phi0 = phiAlpha(xo, ao, p)
    phi1 = phiAlpha(xo, a1, p)
    phip0 = phipAlpha(xo, ao, p)
    print(phi0, phi1, phip0)
    # print("superior")
    # print(phip0*(a1-ao)**2)
    # print("inferior")
    # print(2*(phi1-phi0-(phip0*(a1-ao))))
    return ao - (phip0*(a1-ao)**2)/(2*(phi1-phi0-(phip0*(a1-ao))))


def newton(xo, p, ao, itmax=100, tol=1e-5):
    k = 0
    while abs(phipAlpha(xo, ao, p)) > tol:
        phiap = phipAlpha(xo, ao, p)
        phiapp = phipp(xo, ao, p)
        ak = ao - phiap/phiapp
        print(f"{k}, {ak}")
        k = k+1
        ao = ak
        if k >= itmax:
            print("Iteraciones exdidas")
            break
    return ak




# x = np.linspace(0, 2, 50)
x0List = [2, 3]
x0 = np.array(x0List)

print("Ev Gradiente en 0")

print(grad([0,0]))
print("Ev Hessiano")
print(np.linalg.det(hessiano(x0)))


p = dirgrad(x0)
print("max descenso")
amd = maximoDescenso(x0, p)
print(amd)
print(cw(x0, p, amd, 1e-4, .99))

print("interpolación")
aint = interpolacion(x0, p, 0.5)
print(aint)
print(cw(x0, p, amd, 1e-4, .99))

print("Newton")
newt=newton(x0, p, 0)
print(cw(x0,p,newt,1e-4,0.99))


a = np.linspace(0, 3, 100)
for i in a:
    print(i, cw(x0, p, i, 1e-4, 0.99))


start = timer()
print(start)
xfin = gradDescent(x0)
end = timer()
print(end)
print(f"Evaluacion del mínimo: {grad(xfin)}")
# print(f"Tiempo de ejecución: {end-start} s")
p = dirgrad(x0)
