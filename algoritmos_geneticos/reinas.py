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

def evalReinas(individuo):
    length = len(individuo)
    colisiones = 0

    if len(set(individuo)) != length:
        colisiones += (length - len(set(individuo)))

    for i in range(length):
        for j in range(i+1, length):
            if abs(i - j) == abs(individuo[i] - individuo[j]):
                colisiones += 1
    return colisiones,


toolbox.register('evaluate', evalReinas)
toolbox.register('select', tools.selTournament, tournsize=2)
toolbox.register('mate', tools.cxOnePoint)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)


pop = toolbox.population(n=1000) #Se crea una poblacion de 100 individuos
hof = tools.HallOfFame(1)
stats = tools.Statistics(key=operator.attrgetter('fitness.values'))
stats.register('mean', numpy.mean)
stats.register('min', numpy.min)

result, log = algorithms.eaSimple(pop,
                                  toolbox,
                                  cxpb=0.7,
                                  mutpb=0.2,
                                  ngen=1,
                                  stats=stats,
                                  halloffame=hof,
                                  verbose=False)

best_individual = tools.selBest(result, k=1)[0]
print(evalReinas(best_individual)[0])
print(best_individual)