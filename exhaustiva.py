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


def ecuacion(x):
    return 10 - math.cos(x - 0.1)**2


def derivada(x):
    return 2*math.sin(x-0.1)*math.cos(x-0.1)


def busquedaExhaustiva(h, inferior, superior):
    k = 0
    if h > 0:
        print("empezando por izquierda")
        i = np.arange(inferior, superior, h)
        for paso in i:
            f = ecuacion(paso)
            fprima = derivada(paso)
            print(f'k:{k}   h: %.2f     f(x):%.6f     f\'(x):%.6f' % (paso, f ,fprima))
            k = k + 1
    elif h < 0:
        print("empezando por la derecha")


busquedaExhaustiva(.1, -0.4, 0.64)
