from deap import algorithms
from deap import base
from deap import creator
from deap import tools

import random 
import numpy
import operator

NUM_COLUMNAS_FILAS = 8
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

toolbox.register('attr_perm', random.sample, range(1, NUM_COLUMNAS_FILAS+1), NUM_COLUMNAS_FILAS)
toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.attr_perm)

toolbox.register('population', tools.initRepeat, list, toolbox.individual)

#Compara la cantidad de ataques entre reinas
def evalReinas(individuo):
    length = len(individuo)
    ataques = 0

    if len(set(individuo)) != length: #comprueba si hay reinas en la misma fila
        ataques += (length - len(set(individuo)))

    for i in range(length): #Comprueba las diagonales
        for j in range(i+1, length):
            if abs(i - j) == abs(individuo[i] - individuo[j]):
                ataques += 1
    return ataques, #Cantidad de ataques

def cross(individual1, individual2):
    k = random.randint(3,len(individual1)-3) #Posicion del cruce
    cross1 = [num for num in individual2 if num not in individual1[:k]] #Valores no repetidos para individuo1
    cross2 = [num for num in individual2 if num not in individual2[k:]] #Valores no repetidos para individuo2
    individual_1 = individual1[:k]+cross1
    individual_2 = cross2+individual2[k:]
    for i in range(len(individual1)):
        individual1[i] = individual_1[i]
    for i in range(len(individual2)):
        individual2[i] = individual_2[i]
    return individual1, individual2

def mutation(individual):
    firstPick = random.randint(0, NUM_COLUMNAS_FILAS-1) #selecciona un numero aleatorio
    secondPick = random.randint(0, NUM_COLUMNAS_FILAS-1) #selecciona un numero aleatorio
    while secondPick == firstPick:  # Continuar seleccionando mientras secondPick sea igual a firstPick
        secondPick = random.randint(0, NUM_COLUMNAS_FILAS-1)
    return individual, #retorna la lista



toolbox.register('evaluate', evalReinas)
toolbox.register('select', tools.selTournament, tournsize=2)
toolbox.register('mate', cross) #Utiliza la funcion de cruce 
toolbox.register("mutate", mutation)  #Utiliza la funcion mutation


pop = toolbox.population(n=1000)
hof = tools.HallOfFame(1)
stats = tools.Statistics(key=operator.attrgetter('fitness.values'))
stats.register('mean', numpy.mean)
stats.register('min', numpy.min)

result, log = algorithms.eaSimple(pop,
                                  toolbox,
                                  cxpb=0.7,
                                  mutpb=0.2,
                                  ngen=10,
                                  stats=stats,
                                  halloffame=hof,
                                  verbose=False)

best_individual = tools.selBest(result, k=1)[0]
print("Cantidad de ataques:", evalReinas(best_individual)[0])
print("Tablero:",best_individual)