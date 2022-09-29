# 1. Hacer las programación de los métodos de búsqueda exhaustiva y razón dorada
# f(x) = 10 - cos ^ 2(x-0.1)
# calcular el mínimo en el intervalo[-0.4, 0.65] utilizando los métodos
# b) utilizando razón dorada calcular el mínimo en el intervalo
import math


tol = 1e-6
R = (-1 + 5 ** 0.5) / 2


def ecuacion(x):
    return (-1+x/math.sqrt(5))*math.e**(-(-1+((2*x)/math.sqrt(5)))**2 - (-1+(x/math.sqrt(5)))**2)


def derivada(x):
    return -2*math.cos(x) + x/5


def razonDorada(xl, xu):
    print(f"<=== Calculado con Tol: {tol}  ===>\n")
    k = 0
    i = 1
    d = R*(xu - xl)
    x1 = xl + d
    x2 = xu - d
    fxl = ecuacion(xl)
    fxu = ecuacion(xu)
    fx1 = ecuacion(x1)
    fx2 = ecuacion(x2)
    # print(f'k:{k}   xl:{round(xl, 6)}  f(xl): {round(fxl, 6)}   x2:{round(x2, 6)}  f(x2):{round(fx2, 6)}  x1:{round(x1, 6)}   f(x1):{round(fx1, 6)}   xu:{round(xu, 6)}   f(xu):{round(fxu, 6)}   d:{round(d, 6)}')
    imprimir(xl, xu, k, d, x1, x2, fxl, fxu, fx1, fx2)
    k = k + 1

    while xu-xl > tol:

        if fx1 < fx2:
            xl = x2
            fxl = fx2
            x2 = x1
            fx2 = fx1
            d = d * R
            x1 = xl + d
            fx1 = ecuacion(x1)
            imprimir(xl, xu, k, d, x1, x2, fxl, fxu, fx1, fx2)
        elif fx2 < fx1:
            xu = x1
            fxu = fx1
            x1 = x2
            fx1 = fx2
            d = d * R
            x2 = xu - d
            fx2 = ecuacion(x2)
            imprimir(xl, xu, k, d, x1, x2, fxl, fxu, fx1, fx2)
        k = k+1
        i = i+1
    
    if fx1 < fx2:
        print(
            f"Mínimo encontrado con Tol = {tol}  x: {round( x1 , 6)}   f(x): {round(ecuacion(x1),6)}   f\'(x): {round(derivada(x1),6)}")
    elif fx2 < fx1:
        print(
            f"Mínimo encontrado con Tol = {tol}  x: {round( x2 , 6)}   f(x): {round(ecuacion(x2),6)}   f\'(x): {round(derivada(x2),6)}")


def imprimir(xl, xu, k, d, x1, x2, fxl, fxu, fx1, fx2):
    print(f'k:{k}   xl:{round(xl, 6)}  f(xl): {round(fxl, 6)}   x2:{round(x2, 6)}  f(x2):{round(fx2, 6)}  x1:{round(x1, 6)}   f(x1):{round(fx1, 6)}   xu:{round(xu, 6)}   f(xu):{round(fxu, 6)}   d:{round(d, 6)}')
    if fx1 > fx2:
        print(
            f' x: {round(x1,6)} f(x): {round(ecuacion(x1),6)}   f\'(x): {round(derivada(x1),6)}')
    elif fx2 > fx1:
        print(
            f' x: {round(x2,6)} f(x): {round(ecuacion(x2),6)}   f\'(x): {round(derivada(x2),6)}')


razonDorada(0.75, 1.25)
