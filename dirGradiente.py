import numpy as np
import math


# función original f(x) = x_1 e^(-x_1^2 - x_2^2)


def f(x1, x2):
    """Funición a evaluar"""
    a = (np.e**(-x1**2 - x2**2))*x1
    return a


def grad(x1, x2):
    """ Retorna gradiente  ∇f"""
    list1 = [1-2*x1**2, -2*x1*x2]
    descenso = np.array(list1)*math.e**(-x1**2 - x2**2)
    return descenso


def dirgrad(x1, x2):
    """Retorna la dirección del gradiente p"""
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


def exhaustivo(p, xini, alpha=0, h=0.05,):
    """Busqueda de minimo con metodo exaustivo. puedes cambiar el paso"""
    k = 0
    fantes = f(xini[0], xini[1])
    while True:
        alpha = alpha + h
        x = xini + alpha*p
        fnow = f(x[0], x[1])
        print(k, x[0], x[1], alpha, fnow)
        k = k+1
        if (fnow > fantes):
            return fantes
            break
        fantes = fnow


def razonDorada(aLow, aUp, p, xini, tol=1e-6):
    print(f"<=== Calculado con Tol: {tol}  ===>\n")
    R = (-1 + 5 ** 0.5) / 2
    k = 0
    i = 1
    d = R*(aUp - aLow)
    a1 = aLow + d
    a2 = aUp - d
    faLow = phiAlpha(xini, aLow, p)
    faUp = phiAlpha(xini, aUp, p)
    fa1 = phiAlpha(xini, a1, p)
    fa2 = phiAlpha(xini, a2, p)
    # print(f'k:{k}   aLow:{round(aLow, 6)}  f(aLow): {round(faLow, 6)}   a2:{round(a2, 6)}  f(a2):{round(fa2, 6)}  a1:{round(a1, 6)}   f(a1):{round(fa1, 6)}   aUp:{round(aUp, 6)}   f(aUp):{round(faUp, 6)}   d:{round(d, 6)}')
    # imprimir(aLow, aUp, k, d, a1, a2, faLow, faUp, fa1, fa2)
    k = k + 1

    while aUp-aLow > tol:

        if fa1 < fa2:
            aLow = a2
            faLow = fa2
            a2 = a1
            fa2 = fa1
            d = d * R
            a1 = aLow + d
            fa1 = phiAlpha(xini, a1, p)
            print(aLow, aUp, k, p, a1, a2, k,)
            # imprimir(aLow, aUp, k, d, a1, a2, faLow, faUp, fa1, fa2)
        elif fa2 < fa1:
            aUp = a1
            faUp = fa1
            a1 = a2
            fa1 = fa2
            d = d * R
            a2 = aUp - d
            fa2 = phiAlpha(xini, a2, p)
        k = k+1
    if fa1 < fa2:
        print(
            f"Mínimo encontrado con Tol = {tol}  x: {round( a1 , 6)}   f(x): {round(phiAlpha(xini,a1,p),6)}    ")
    elif fa2 < fa1:
        print(
            f"Mínimo encontrado con Tol = {tol}  x: {round( a2 , 6)}   f(x): {round(phiAlpha(xini,a2,p),6)}    ")


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


def newton(xo, p, ao, itmax=10, tol=1e-6):
    k = 0
    while abs(phipAlpha(xo, ao, p)) > tol:
        phiap = phipAlpha(xo, ao, p)
        phiapp = phipp(xo, ao, p)
        ak = ao - phiap/phiapp
        print(k, ak, phiap, phiapp, ao)
        k = k+1
        ao = ak
        if k >= itmax:
            print("Iteraciones exdidas")
            break


def interpolacion(xo, p, a1, ao=0):
    phi0 = phiAlpha(xo, ao, p)
    phi1 = phiAlpha(xo, a1, p)
    phip0 = phipAlpha(xo, ao, p)
    return ao - (phip0*(a1-ao)**2)/(2*(phi1-phi0-(phip0*(a1-ao))))


def maximoDescenso(xini):
    """Regresa alpha maximo descenso """
    vgrad = grad(xini[0], xini[1])
    amaxDes = -np.dot(vgrad, p) / \
        np.dot(np.dot(hessiano(xini[0], xini[1]), p), p)
    return amaxDes


xini = [-1, -1]
p = dirgrad(xini[0], xini[1])
# print("<==Exhaustivo==>")
# a = exhaustivo(p, xini)
# print(a)
# print("<==Razón Dorada==>")
# razonDorada(0, 2, p, xini)
print("<==Metodo de Newton==>")
print(newton(xini, p, 0))
# print("<==Iterpolación==>")
# print(interpolacion(xini, p, 0.3))
