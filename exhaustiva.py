# 1. Hacer las programación de los métodos de búsqueda exhaustiva y razón dorada
# f(x) = 10 - cos ^ 2(x-0.1)
# calcular el mínimo en el intervalo[-0.4, 0.65] utilizando los métodos
# a) búsqueda exhaustiva con x ^ (0)=-0.4  y h=0.1
# c) cuantas iteraciones de búsqueda exhaustiva son necesaria para tener un resultado equivalente al resultado de razón dorada
import numpy as np
import math
import pprint


tabla = []
miDiccionario = {}


def ecuacion(x):
    return -2*math.sin(x) + x**2/10


def derivada(x):
    return -2*math.cos(x) + x/5


def busquedaExhaustiva(h, inferior, superior):
    k = 0
    if h > 0:
        print("<=== Empezando por izquierda ==>\n")
        i = np.arange(inferior, superior, h)
        evaluar(k, i)
    elif h < 0:
        print("<=== Empezando por la derecha ===>\n")
        i = np.arange(superior, inferior, h)
        evaluar(k, i)


def evaluar(k, i):
    for paso in i:
        # Evaluamos en f(x) y en f´(x)
        f = ecuacion(paso)
        fprima = derivada(paso)
        print(f'k:{k}   x: {round(paso,6)}     f(x):{round(f,6)}    f\'(x):{round(fprima,6)}')
        # Almacenamos valores en Array
        miDiccionario[k] = round(fprima, 6)
        tabla.append({'k': k, 'x': round(paso, 2),
                     'fx': round(f, 6), 'fp': round(fprima, 6)})
        k = k + 1


busquedaExhaustiva(.01, 0, 2.01)
min = sorted(miDiccionario.items(), key=lambda x: abs(0 - x[1]))
# imprimir los valores por cercanía a 0
# pprint.pprint(min)
# print(min[0][1])

# pprint.pprint(sorted(tabla, key=lambda x: x['f',0], reverse=True))
print(
    f'\nMínimo encontrado en la iteración: {min[0][0]} \nCon los siguientes valores: ')
# accedemos al Array con el valor minimo de midiccionario
pprint.pprint(tabla[min[0][0]])
