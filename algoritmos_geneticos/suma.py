from deap import algorithms
from deap import base
from deap import creator
from deap import tools

import random 
import numpy
import operator

MAX_INT = 8 #Cantidad de individuos que se van a crear
creator.create("FitnessMin", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

toolbox.register('attr_int', random.randint, 0, 7) #Se crea un individuo con un valor aleatorio entre 0 y 7
toolbox.register('individual', tools.initRepeat,
                 creator.Individual, toolbox.attr_int, MAX_INT)#Se crean varios individuos
toolbox.register('population', tools.initRepeat, list, toolbox.individual)#Se crea una poblacion de individuos

def eval(individual):
    return sum(individual),

toolbox.register('evaluate', eval)
toolbox.register('select', tools.selTournament, tournsize=2)
toolbox.register('mate', tools.cxOnePoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)


pop = toolbox.population(n=100)
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
print( int(eval(best_individual)[0]))