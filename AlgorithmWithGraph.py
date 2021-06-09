import random
import time

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.animation as animation

# константы генетического алгоритма
POPULATION_SIZE = 100  # количество индивидуумов в популяции
MAX_GENERATIONS = 200  # максимальное количество поколений

P_CROSSOVER = 0.9  # вероятность скрещивания
P_MUTATION = 0.1  # вероятность мутации индивидуума

N_VECTOR = 2  # количество генов в хромосоме

LIMIT_VALUE_TOP = 100
LIMIT_VALUE_DOWN = -100

RANDOM_SEED = 1
random.seed(RANDOM_SEED)


class Individual(list):
    def __init__(self, *args):
        super().__init__(*args)
        self.value = 0


def fitness_function(f):
    return f[0] ** 2 + 1.5 * f[1] ** 2 - 2 * f[0] * f[1] + 4 * f[0] - 8 * f[1]


def individualCreator():
    return Individual([random.randint(LIMIT_VALUE_DOWN, LIMIT_VALUE_TOP) for i in range(N_VECTOR)])


def populationCreator(n=0):
    return list([individualCreator() for i in range(n)])


population = populationCreator(n=POPULATION_SIZE)

fitnessValues = list(map(fitness_function, population))

for individual, fitnessValue in zip(population, fitnessValues):
    individual.value = fitnessValue

MinFitnessValues = []
meanFitnessValues = []
BadFitnessValues = []

population.sort(key=lambda ind: ind.value)
print(population)


def clone(value):
    ind = Individual(value[:])
    ind.value = value.value
    return ind


def selection(popula, n=POPULATION_SIZE):
    offspring = []
    for i in range(n):
        i1 = i2 = i3 = i4 = 0
        while i1 in [i2, i3, i4] or i2 in [i1, i3, i4] or i3 in [i1, i2, i4] or i4 in [i1, i2, i3]:
            i1, i2, i3, i4 = random.randint(0, n - 1), random.randint(0, n - 1), random.randint(0,
                                                                                                n - 1), random.randint(
                0, n - 1)

        offspring.append(
            min([popula[i1], popula[i2], popula[i3], popula[i4]], key=lambda ind: ind.value))

    return offspring


def crossbreeding(object_1, object_2):
    s = random.randint(1, len(object_1) - 1)
    object_1[s:], object_2[s:] = object_2[s:], object_1[s:]


def mutation(mutant, indpb=0.04, percent=0.05):
    for index in range(len(mutant)):
        if random.random() < indpb:
            mutant[index] += random.randint(-1, 1) * percent * mutant[index]


generationCounter = 0


def animate(i):
    global generationCounter

    ax.clear()
    ax.plot_wireframe(X, Y, Z, rstride=30, cstride=30)
    ax.text(LIMIT_VALUE_DOWN, LIMIT_VALUE_DOWN, 40000, "Поколение: " + str(generationCounter))

    ax.scatter(2, 4, -12, marker='o', edgecolors='green', linewidths=4)

    for individ in population:
        ax.scatter(individ[0], individ[1], individ.value, marker='*', edgecolors='red')

    if generationCounter == 1:
        time.sleep(3)

    generationCounter += 1
    offspring = selection(population)
    offspring = list(map(clone, offspring))

    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < P_CROSSOVER:
            crossbreeding(child1, child2)

    for mutant in offspring:
        if random.random() < P_MUTATION:
            mutation(mutant, indpb=1.0 / N_VECTOR)

    freshFitnessValues = list(map(fitness_function, offspring))

    for individual, fitnessValue in zip(offspring, freshFitnessValues):
        individual.value = fitnessValue

    population[:] = offspring

    fitnessValues = [ind.value for ind in population]

    minFitness = min(fitnessValues)
    meanFitness = sum(fitnessValues) / len(population)
    maxFitness = max(fitnessValues)
    MinFitnessValues.append(minFitness)
    meanFitnessValues.append(meanFitness)
    BadFitnessValues.append(maxFitness)

    plt.plot(MinFitnessValues[int(MAX_GENERATIONS * 0.04):], color='red')
    plt.plot(meanFitnessValues[int(MAX_GENERATIONS * 0.04):], color='green')
    plt.plot(BadFitnessValues[int(MAX_GENERATIONS * 0.04):], color='blue')
    print(
        f"Поколение {generationCounter}: Функция приспособленности. = {minFitness}, Средняя приспособ.= {meanFitness}")

    best_index = fitnessValues.index(min(fitnessValues))
    print("Лучший индивидуум = ", *population[best_index], "\n")


fig = plt.figure()
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax2 = fig.add_subplot(1, 2, 2)

X = np.arange(LIMIT_VALUE_DOWN, LIMIT_VALUE_TOP, 0.5)
Y = np.arange(LIMIT_VALUE_DOWN, LIMIT_VALUE_TOP, 0.5)
X, Y = np.meshgrid(X, Y)
Z = (X ** 2 + 1.5 * Y ** 2 - 2 * X * Y + 4 * X - 8 * Y)

ani = animation.FuncAnimation(fig, animate, interval=2000)


plt.xlabel('Поколение')
plt.ylabel('Мин/средняя/max приспособленность')

plt.title("Нахождение глобального минимума фукнции с помощью генетического алгоритма")
plt.show()

