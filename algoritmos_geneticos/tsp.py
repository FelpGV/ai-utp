from deap import algorithms
from deap import base
from deap import creator
from deap import tools

import random 
import numpy
import operator

#ulysses16
tsp = [
    [0, 509, 501, 312, 1019, 736, 656, 60, 1039, 726, 2314, 479, 448, 479, 619, 150],
    [509, 0, 126, 474, 1526, 1226, 1133, 532, 1449, 1122, 2789, 958, 941, 978, 1127, 542],
    [501, 126, 0, 541, 1516, 1184, 1084, 536, 1371, 1045, 2728, 913, 904, 946, 1115, 499],
    [312, 474, 541, 0, 1157, 980, 919, 271, 1333, 1029, 2553, 751, 704, 720, 783, 455],
    [1019, 1526, 1516, 1157, 0, 478, 583, 996, 858, 855, 1504, 677, 651, 600, 401, 1033],
    [736, 1226, 1184, 980, 478, 0, 115, 740, 470, 379, 1581, 271, 289, 261, 308, 687],
    [656, 1133, 1084, 919, 583, 115, 0, 667, 455, 288, 1661, 177, 216, 207, 343, 592],
    [60, 532, 536, 271, 996, 740, 667, 0, 1066, 759, 2320, 493, 454, 479, 598, 206],
    [1039, 1449, 1371, 1333, 858, 470, 455, 1066, 0, 328, 1387, 591, 650, 656, 776, 933],
    [726, 1122, 1045, 1029, 855, 379, 288, 759, 328, 0, 1697, 333, 400, 427, 622, 610],
    [2314, 2789, 2728, 2553, 1504, 1581, 1661, 2320, 1387, 1697, 0, 1838, 1868, 1841, 1789, 2248],
    [479, 958, 913, 751, 677, 271, 177, 493, 591, 333, 1838, 0, 68, 105, 336, 417],
    [448, 941, 904, 704, 651, 289, 216, 454, 650, 400, 1868, 68, 0, 52, 287, 406],
    [479, 978, 946, 720, 600, 261, 207, 479, 656, 427, 1841, 105, 52, 0, 237, 449],
    [619, 1127, 1115, 783, 401, 308, 343, 598, 776, 622, 1789, 336, 287, 237, 0, 636],
    [150, 542, 499, 455, 1033, 687, 592, 206, 933, 610, 2248, 417, 406, 449, 636, 0]
]

num_cities = len(tsp)

#Crea un individuo como permutacion de num_cities sin que coincidan indice y valor
def perm():
    individual = random.sample(range(num_cities), num_cities) #permutacion para crear individuo
    while any(individual[i] == i for i in range(len(individual))): #verifica coincidencia de valor e indice hasta encontrar un individuo correcto
        individual = perm()
    return individual

# print(num_cities)
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# toolbox.register('perm', perm)
toolbox.register('individual', tools.initIterate, creator.Individual, perm)
toolbox.register('population', tools.initRepeat, list, toolbox.individual)

#Suma las distancias
def evalTSP(individual):
    distance = 0
    for i in range(num_cities):
        distance += tsp[i][individual[i]]
    return distance,

def cross(individual1, individual2):
    k = random.randint(3,len(individual1)-3) #Posicion del cruce
    cross1 = [num for num in individual2 if num not in individual1[:k]] #Valores no repetidos para individuo1
    cross2 = [num for num in individual2 if num not in individual2[k:]] #Valores no repetidos para individuo2
    individual_1 = individual1[:k]+cross1 #nuevo individuo1
    individual_2 = cross2+individual2[k:] #nuevo individuo2
    while any(individual_1[i] == i for i in range(len(individual1))): #verifica coincidencia de valor e indice hasta encontrar un individuo correcto
        random.shuffle(cross1) #Reorganiza de cross1
        individual_1 = individual1[:k]+cross1
    while any(individual_2[i] == i for i in range(len(individual1))): #verifica coincidencia de valor e indice hasta encontrar un individuo correcto
        random.shuffle(cross2) #Reorganiza de cross2
        individual_2 = cross2+individual2[k:]
    for i in range(len(individual1)):
        individual1[i] = individual_1[i]
    for i in range(len(individual2)):
        individual2[i] = individual_2[i]
     
    return individual1, individual2

def mutation(individual):
    firstPick = random.randint(0, num_cities-1) #selecciona un numero aleatorio
    secondPick = random.randint(0, num_cities-1) #selecciona un numero aleatorio
    while secondPick == firstPick:  # Continuar seleccionando mientras secondPick sea igual a firstPick
        secondPick = random.randint(0, num_cities-1)
    if individual[firstPick] != secondPick and individual[secondPick] != firstPick: 
        individual[firstPick], individual[secondPick] = individual[secondPick], individual[firstPick] #intercambia los valores de firstPick y secondPick
    return individual, #retorna la lista

toolbox.register('evaluate', evalTSP)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mutate", mutation) #Utiliza la funcion mutation
toolbox.register("mate", cross) #Utiliza la funcion de cruce 

pop = toolbox.population(n=100)
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
print("Mejor distancia:",evalTSP(best_individual)[0])
print("Ruta:",best_individual)