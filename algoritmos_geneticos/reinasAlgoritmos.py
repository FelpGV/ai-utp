import random

radomQueens = list(range(1, 9))
random.shuffle(radomQueens)

def cross(individual):
    secondCut = individual[3:8]#corte desde la posicion 3
    random.shuffle(secondCut) #se mezcla el corte
    return individual[0:3] + secondCut #retorna el corte con el resto de la lista

def mutation(individual):
    firstPick = random.randint(0, 7) #selecciona un numero aleatorio entre 0 y 7
    secondPick = random.randint(0, 7) #selecciona un numero aleatorio entre 0 y 7
    while secondPick == firstPick:  # Continuar seleccionando mientras secondPick sea igual a firstPick
        secondPick = random.randint(0, 7)
    individual[firstPick], individual[secondPick] = individual[secondPick], individual[firstPick] #intercambia los valores de firstPick y secondPick
    return individual, individual[firstPick], individual[secondPick] #retorna la lista, el valor de firstPick y el valor de secondPick

def evalQueens(individual):
    length = len(individual)
    colisiones = 0
    for i in range(length):
        for j in range(i+1, length):
            if abs(i - j) == abs(individual[i] - individual[j]):
                colisiones += 1
    return colisiones


print(radomQueens)
crossQueens = cross(radomQueens)
print(crossQueens)
mutatedQueens = mutation(crossQueens)
print(mutatedQueens)

while(evalQueens(mutatedQueens[0]) > 0):
    crossQueens = cross(mutatedQueens[0])
    mutatedQueens = mutation(mutatedQueens[0])
    print(mutatedQueens)