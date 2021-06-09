import random
import matplotlib.pyplot as plt

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
print([str(ind) + ", " + str(ind.value) for ind in population])


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

while generationCounter < MAX_GENERATIONS:
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
    print(
        f"Поколение {generationCounter}: Функция приспособленности. = {minFitness}, Средняя приспособ.= {meanFitness}")
    best_index = fitnessValues.index(min(fitnessValues))
    print("Лучший индивидуум = ", *population[best_index], "\n")

plt.plot(MinFitnessValues[int(MAX_GENERATIONS * 0.1):], color='red')
plt.plot(meanFitnessValues[int(MAX_GENERATIONS * 0.1):], color='green')
plt.plot(BadFitnessValues[int(MAX_GENERATIONS * 0.1):], color='blue')
plt.xlabel('Поколение')
plt.ylabel('Мин/средняя/max приспособленность')
plt.title('Зависимость min, mean, max приспособленности от поколения')
plt.show()
