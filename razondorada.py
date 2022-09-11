# 1. Hacer las programación de los métodos de búsqueda exhaustiva y razón dorada
# 2. Para la función
# f(x) = 10 - cos ^ 2(x-0.1)
# calcular el mínimo en el intervalo[-0.4, 0.65] utilizando los métodos
# a) búsqueda exhaustiva con x ^ (0)=-0.4  y h=0.1
# b) utilizando razón dorada calcular el mínimo en el intervalo
# c) cuantas iteraciones de búsqueda exhaustiva son necesaria para tener un resultado equivalente al resultado de razón dorada
# En todos los casos reportar las iteraciones. así como los valores de x, f(x) y f'(x) en cada iteración.
import numpy as np
import math
import pprint


tabla = []
miDiccionario = {}
tol = 1e-6
R = (-1 + 5 ** 0.5) / 2


def ecuacion(x):
    # return 10 - math.cos(x - 0.1)**2
    return 2*math.sin(x) - (x**2)/10


def derivada(x):
    # return 2*math.sin(x-0.1)*math.cos(x-0.1)
    return x**3-13


def razonDorada(xl, xu):
    k = 0
    i = 1
    d = R*(xu - xl)
    x1 = xl + d
    x2 = xu - d
    fxl = ecuacion(xl)
    fxu = ecuacion(xu)
    fx1 = ecuacion(x1)
    fx2 = ecuacion(x2)
    print(f'k:{k}   xl:{round(xl, 6)}  f(xl): {round(fxl, 6)}   x2:{round(x2, 6)}  f(x2):{round(fx2, 6)}  x1:{round(x1, 6)}   f(x1):{round(fx1, 6)}   xu:{round(xu, 6)}   f(xu):{round(fxu, 6)}   d:{round(d, 6)}')
    tabla.append({'k': k, 'xl': round(xl, 6), 'fxl': round(fxl, 6),  'x2': round(x2, 6),  'fx2': round(
        fx2, 6),  'x1': round(x1, 6),  'fx1': round(fx1, 6),  'xu': round(xu, 6),  'd': round(d, 6), })
    k = k + 1

    # TODO: cambiar por tol
    while i < 20:

        if fx1 > fx2:
            xl = x2
            fxl = fx2
            x2 = x1
            fx2 = fx1
            d = d * R
            x1 = xl + d
            fx1 = ecuacion(x1)
            imprimir(xl, xu, k, d, x1, x2, fxl, fxu, fx1, fx2)
        elif fx2 > fx1:
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


def imprimir(xl, xu, k, d, x1, x2, fxl, fxu, fx1, fx2):
    print(f'k:{k}   xl:{round(xl, 6)}  f(xl): {round(fxl, 6)}   x2:{round(x2, 6)}  f(x2):{round(fx2, 6)}  x1:{round(x1, 6)}   f(x1):{round(fx1, 6)}   xu:{round(xu, 6)}   f(xu):{round(fxu, 6)}   d:{round(d, 6)}')
    if fx1 > fx2:
        print(derivada())
    elif fx2 > fx1:
        print(derivada())   
    tabla.append({'k': k, 'xl': round(xl, 6), 'fxl': round(fxl, 6),  'x2': round(x2, 6),  'fx2': round(
        fx2, 6),  'x1': round(x1, 6),  'fx1': round(fx1, 6),  'xu': round(xu, 6),  'd': round(d, 6), })


razonDorada(0, 4)
# pprint.pprint(tabla)
